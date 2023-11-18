import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

#通常是用卡尔曼滤波器进行目标状态的预测，以提供下一时刻的估计
    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    #这个方法的作用似乎是在一次性对多个轨迹进行卡尔曼滤波器的预测，并更新其状态。
    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet
        这个方法的作用是初始化并激活一个新的轨迹，设置轨迹的初始状态、ID、起始帧等信息"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """重新激活一个已有的轨迹，可能是因为轨迹在一段时间内失去了目标并再次找到"""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        # 这个方法的作用是更新匹配到的轨迹信息，
        # 通常在目标跟踪过程中，用于将当前观测到的目标信息与先前的轨迹信息进行更新。
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        # 用于将边界框位置信息从(top left x, top left y, width, height)格式
        # 转换为(center x, center y, aspect ratio, height)格式
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    # 边界框位置信息从(min x, min y, max x, max y)格式
    # 转换为(top left x, top left y, width, height)格式
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        #__repr__ 方法是一个特殊的方法，用于返回对象的“官方”字符串表示形式。
        # 在这个特定的实现中，它返回一个字符串，表示轨迹对象的一些关键信息。
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30): # 默认为30帧/秒
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size # 其值为缓冲区大小
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = [] # type: list[STrack], 用于存储激活的轨迹
        refind_stracks = [] # type: list[STrack], 用于存储重新找到的轨迹
        lost_stracks = [] # type: list[STrack], 用于存储丢失的轨迹
        removed_stracks = [] # type: list[STrack], 用于存储移除的轨迹

        # 解析输出结果中的检测框和分数
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale # 将检测框的坐标缩放到原始图像尺寸

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high) # inds_second 用于存储低分检测框的索引
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack], 用于存储跟踪的轨迹
        for track in self.tracked_stracks:
            # 将未激活的轨迹对象添加到 unconfirmed 列表，
            # 已激活的轨迹对象添加到 tracked_stracks 列表。
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        # 确保了所有已跟踪的轨迹和已丢失的轨迹都在同一个轨迹池中
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        # 对轨迹池中的所有轨迹进行多目标卡尔曼滤波器的预测，以更新其状态
        STrack.multi_predict(strack_pool)
        #  计算轨迹池中的轨迹与当前帧检测到的目标框 detections 之间的IoU距离
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
        # 如果不是MOT20数据集，则调用 matching.fuse_score 函数将IoU距离与检测分数进行融合
            dists = matching.fuse_score(dists, detections)
        # 使用匈牙利算法进行线性分配，得到匹配的轨迹与检测框的索引。matches 存储匹配的索引对，
        # u_track 存储未匹配的轨迹索引，u_detection 存储未匹配的检测框索引。
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        # 处理了匹配成功的轨迹与检测框的情况
        for itracked, idet in matches:
            track = strack_pool[itracked] # 获取轨迹池中的匹配成功的轨迹
            det = detections[idet] # 获取当前帧检测到的匹配成功的目标框
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        # 在第二轮关联中，处理未匹配的已跟踪轨迹与低分数检测框的情况，
        # 并更新轨迹的状态和位置。
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        # 从轨迹池中选取未匹配的已跟踪轨迹（状态为 Tracked)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 处理在第二轮关联中未匹配的已跟踪轨迹，并将它们标记为已丢失
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        # 处理未激活的轨迹，更新其状态和位置，并将其标记为已激活或已移除
        # 这样可以有效处理那些只有一个起始帧的轨迹
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        # 处理了第一轮关联中未匹配的检测框，
        # 对于那些分数超过阈值self.det_thresh 的检测框，创建新的轨迹并将其激活
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        # 处理了已丢失的轨迹
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        # 从已跟踪的轨迹列表中筛选出状态为 Tracked 的轨迹，删除状态不为 Tracked 的轨迹
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        # 将已跟踪的轨迹列表与本帧新激活的轨迹列表合并，确保不重复添加相同 track_id 的轨迹
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        # 将已跟踪的轨迹列表与重新找回的轨迹列表合并，确保不重复添加相同 track_id 的轨迹
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        #  从已丢失的轨迹列表中移除已跟踪的轨迹
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        # 从已丢失的轨迹列表中移除需移除的轨迹,.extend() 方法用于在列表末尾一次性追加另一个序列中的多个值
        self.lost_stracks.extend(lost_stracks)
        # 从已丢失的轨迹列表中移除需移除的轨迹
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        # 移除已跟踪的轨迹列表和已丢失的轨迹列表中重复的轨迹
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        # 获取已跟踪的轨迹中已激活的轨迹，作为最终的输出轨迹列表

        return output_stracks


def joint_stracks(tlista, tlistb):
    # 这个函数的作用是将两个轨迹列表合并成一个，
    # 确保不重复添加相同 track_id 的轨迹。
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    # 这个函数的作用是从第一个轨迹列表
    # 中去除与第二个轨迹列表中相同 track_id 的轨迹。
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    # 这个函数的作用是从两个轨迹列表中去除相同 track_id 的轨迹。
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
