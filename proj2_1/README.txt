1. complementary_filer.py
	the instant rotation matrix is calculated by using scipy.linalg.expm(), which calculate the matrix exponential. This is not simply taking the exponential of the matrix element wisely. 
	To deal with data from 6 axis IMU.
	First, from gyro measurement, integrate angular velocity over time, and calculate the matrix exponential to get the current rotation. Use the initial rotation times current rotation to obtain the orientation of the platform w.r.p to the world frame.
	Secondly, use the accelerometer measurement to calculate the approximate gravity direction w.r.p to the world frame. This is obtained by rotate the linear acceleration w.r.p to the body frame to the world frame by the rotation matrix, then normalize it. 
	Thirdly, calculate the correction rotation to fix the approximate gravity direction. Need to calculate norm of acceleration measurement, compare it with unit gravity acceleration. If the difference is big, it means the platform is not at rest, and the correction rotation might be changed by a smaller weight. 
	Finally, return the final estimate rotation of the platform w.r.t to world frame.



2. estimate_post_ransac.py

def solve_w_t(uvd1, uvd2, R0)
	each correspondence would create two linear equation. Loop through each correspondence(n) to form 2*n equations. use least square problem to solve the overconstrained linear system. 



def find_inliers(w, t, uvd1, uvd2. R0, threshold)
	use the estimated angular velocity, translation, uvd1, uvd2 and initial Rotation to calculate the delta, which is the difference between the actual and estimated value. Use threshold to determine which are inliers. 