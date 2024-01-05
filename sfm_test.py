import os
import argparse
import cv2
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

# Step I
def SIFT(img_list):
	img_kp_list = list()
	img_des_list = list()
	matche_list = list()

	sift_init = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10)
	for img_idx in img_list:
		if img_idx is not None:
			key_points, descriptor = sift_init.detectAndCompute(img_idx, None)
			img_kp_list.append(key_points)
			img_des_list.append(descriptor)
		else:
			print("no image")
			sys.stdout.flush()

	for des_idx in range(len(img_des_list)-1):
		bf = cv2.BFMatcher(cv2.NORM_L2) # Brute-Force Search
		matches = bf.knnMatch(img_des_list[des_idx], img_des_list[des_idx+1], k=2) # K-Nearest Neighbors
		matches_good = [m for m, n in matches if m.distance < 0.7*n.distance]
		matche_list.append(matches_good)
		#sorted_matches = sorted(matches_good, key=lambda x: x.distance)
		#matches_result = cv2.drawMatches(img_list[des_idx], img_kp_list[des_idx], img_list[des_idx+1], img_kp_list[des_idx+1], sorted_matches, img_list[des_idx+1], flags=2)
		# drwa matches
		#plt.figure(figsize=(12, 12))
		#plt.imshow(matches_result)
		#plt.show()

	return img_kp_list, matche_list

# Step II
def Essential_me(K, img_kp_list, maches_list):
	p1 = np.float32([img_kp_list[0][m.queryIdx].pt for m in maches_list[0]])
	p2 = np.float32([img_kp_list[1][m.trainIdx].pt for m in maches_list[0]])

	focal_length = 1698.873755
	principle_point = (971.7497705, 647.7488275)
	E, mask = cv2.findEssentialMat(np.array(p1), np.array(p2), focal_length, principle_point, cv2.RANSAC, 0.999, 1.0)
	_, R, T, mask = cv2.recoverPose(E, np.array(p1), np.array(p2), K, mask)

	p1_re = list()
	p2_re = list()
	for mask_idx in range(len(mask)):
		if mask[mask_idx] > 0:
			p1_re.append(p1[mask_idx])
			p2_re.append(p2[mask_idx])

	return p1_re, p2_re, R, T, mask

# Step III & IV
def Essential_md(K, R, T, P1, P2, init_flag, R_=0, T_=0):
	R0t = np.zeros((3, 4))
	if init_flag:
		R0t = np.hstack((np.eye(3), np.zeros((3, 1))))
	else:
		R0t[0:3, 0:3] = np.float32(R_)
		R0t[:, 3] = np.float32(T_.T)
	R1t = np.zeros((3, 4))
	R1t[0:3, 0:3] = np.float32(R)
	R1t[:, 3] = np.float32(T.T)
	K_re = np.float32(K)
	points2d_1 = np.dot(K_re, R0t)
	points2d_2 = np.dot(K_re, R1t)
	points3d = cv2.triangulatePoints(points2d_1, points2d_2, np.array(P1).T, np.array(P2).T)

	points3d_list = list()
	rows, cols = points3d.shape
	for p_idx in range(cols):
		points3d_col = points3d[:, p_idx]
		points3d_col /= points3d_col[3]
		points3d_list.append([points3d_col[0], points3d_col[1], points3d_col[2]])

	return points3d_list

# Step IV
def Triangulation(points3d_list, img_kp_list, maches_list, mask):
	matche3d_idx = list()
	for kp_idx in img_kp_list:
		matche3d_idx.append(np.ones(len(kp_idx)) *- 1)

	idx = 0
	for m_idx in range(len(maches_list[0])):
		if mask[m_idx] != 0:
			matche3d_idx[0][int(maches_list[0][m_idx].queryIdx)] = idx
			matche3d_idx[1][int(maches_list[0][m_idx].queryIdx)] = idx
			idx += 1

	return matche3d_idx

# Step V
def Growing_step(K, R, T, maches_list, matche3d_idx, points3d_list, img_kp_list):
	rotation_vec = [np.eye(3, 3), R]
	motion_vec = [np.zeros((3, 1)), T]


	for gs_idx in range(1, len(maches_list)):
		points_3D = list()
		points_2D = list()
		for match in maches_list[gs_idx]:
			query_idx = match.queryIdx
			train_idx = match.trainIdx
			points_3D.append(points3d_list[int(matche3d_idx[gs_idx][query_idx])])
			points_2D.append(img_kp_list[gs_idx + 1][train_idx].pt)
		_, PnP_R, PnP_T, _ = cv2.solvePnPRansac(np.array(points_3D), np.array(points_2D), K, np.array([]))
		rf, _ = cv2.Rodrigues(PnP_R)
		rotation_vec.append(rf)
		motion_vec.append(PnP_T)
		p1 = np.float32([img_kp_list[gs_idx][m.queryIdx].pt for m in maches_list[gs_idx]])
		p2 = np.float32([img_kp_list[gs_idx+1][m.trainIdx].pt for m in maches_list[gs_idx]])
		ohter_points3d_list = Essential_md(K, rf, PnP_T, p1, p2, False, rotation_vec[gs_idx], motion_vec[gs_idx])
		# mixed point
		for m_idx in range(len(maches_list[gs_idx])):
			query_idx_ = maches_list[gs_idx][m_idx].queryIdx
			train_idx_ = maches_list[gs_idx][m_idx].trainIdx
			matche3d_idx_ = matche3d_idx[gs_idx][query_idx_]
			if matche3d_idx_ >= 0:
				matche3d_idx[gs_idx+1][train_idx_] = matche3d_idx_
			points3d_list = np.append(points3d_list, [ohter_points3d_list[m_idx]], axis = 0)
			matche3d_idx[gs_idx][query_idx_] = matche3d_idx[gs_idx+1][train_idx_] = len(points3d_list) - 1

	# bundle adjustment
	for rot_idx in range(len(rotation_vec)):
		rf_re, _ = cv2.Rodrigues(rotation_vec[rot_idx])
		rotation_vec[rot_idx] = rf_re
	for m_re_idx in range(len(matche3d_idx)):
		point3d_idxs = matche3d_idx[m_re_idx]
		key_points = img_kp_list[m_re_idx]
		for p_idx in range(len(point3d_idxs)):
			P, X = cv2.projectPoints(points3d_list[int(point3d_idxs[p_idx])].reshape(1, 1, 3), rotation_vec[m_re_idx], motion_vec[m_re_idx], K, np.array([]))
			P = P.reshape(2)
			err = key_points[p_idx].pt - P
			if abs(err[0]) > 0.5 or abs(err[1]) > 1.0:
				points3d_list[int(point3d_idxs[p_idx])] = None
	i = 0
	while i < len(points3d_list):
		if math.isnan(points3d_list[i][0]):
			points3d_list = np.delete(points3d_list, i, 0)
			i -= 1
		i += 1

	print(len(points3d_list))
	print(len(motion_vec))

	return points3d_list


def Draw_3D(points3d_list):
    mlab.points3d(points3d_list[:, 0], points3d_list[:, 1], points3d_list[:, 2], mode = 'point', name = 'dinosaur')
    mlab.show()


def main(args):
	img_list = list()
	for (path, dir, files) in os.walk(args.path):
		for filename in files:
			ext = os.path.splitext(filename)[-1]
			if ext == '.JPG' or ext == '.PNG' or ext == '.jpg' or ext == '.png':
				full_path = path + filename
				img_rgb = cv2.imread(full_path)
				img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
				img_list.append(img_gray)
				print(full_path)

	K = np.array([
        [1698.873755, 0, 971.7497705],
        [0, 1698.8796645, 647.7488275],
        [0, 0, 1]])

	img_kp_list, maches_list = SIFT(img_list)
	p1_re, p2_re, R, T, mask = Essential_me(K, img_kp_list, maches_list)
	points3d_list = Essential_md(K, R, T, p1_re, p2_re, True)
	matche3d_idx = Triangulation(points3d_list, img_kp_list, maches_list, mask)
	points3d_final = Growing_step(K, R, T, maches_list, matche3d_idx, points3d_list, img_kp_list)
	Draw_3D(points3d_final)

	#ori_pc = pcl.PointCloud(points3d_final) # numpy to pcl
	#pcl.save(ori_pc,"C:/Users/yuncs/Desktop/ori_cloud.pcd")
	


def parse_args():
    parser = argparse.ArgumentParser(description='WKIT')
    parser.add_argument('--path', type = str, default = '', help = 'set of images path')
    args = parser.parse_args()
    print(args)
    return args

if __name__ == '__main__':
	try:
		args = parse_args()
		main(args)
	except Exception:
		print(f'{traceback.format_exc()}')
		sys.stdout.flush()
	finally:
		print('SFM exit')
		sys.stdout.flush()