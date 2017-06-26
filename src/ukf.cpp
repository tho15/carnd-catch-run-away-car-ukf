#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	n_x_ = 5;
	n_aug_ = 7;  // n_x_ +2

	// initial state vector
	x_ = VectorXd(n_x_);
	
	// initial covariance matrix
	P_ = MatrixXd(n_x_, n_x_);

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 0.8;

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 0.6;

	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.15;

	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.15;
	
	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.3;

	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.03;

	// Radar measurement noise standard deviation radius change in m/s
	std_radrd_ = 0.3;
	
	/**
	Complete the initialization. See ukf.h for other member properties.
	Hint: one or more values initialized above might be wildly off...
	*/
	is_initialized_ = false;
	P_ << 1, 0, 0, 0, 0,
		 0, 1, 0, 0, 0,
		 0, 0, 1, 0, 0,
		 0, 0, 0, 1, 0,
		 0, 0, 0, 0, 1;

    // 
	double kappa = 0.0, beta = 2.0, alpha = 0.001;
	
	//kappa = 0.0; beta = 2.0; alpha = 0.001;	
	weights_m_ = VectorXd(2*n_aug_ +1);
	weights_c_ = VectorXd(2*n_aug_ +1);	
	
	lambda_ = alpha*alpha*(n_aug_ + kappa) - n_aug_;
	weights_m_(0) = lambda_/(n_aug_ + lambda_);
	weights_c_(0) = lambda_/(n_aug_ + lambda_) + (1-alpha*alpha+beta);
	for(int i = 1; i < 2*n_aug_+1; i++) {
		weights_m_(i) = 1.0/(2*(n_aug_+lambda_));
		weights_c_(i) = weights_m_(i);
	}
}

UKF::~UKF() {}


void UKF::InitMeasurement(MeasurementPackage meas_package)
{
	time_us_ = meas_package.timestamp_;
	  
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
		/**
		 Convert radar from polar to cartesian coordinates and initialize state.
		 */
		double px = meas_package.raw_measurements_[0]*std::cos(meas_package.raw_measurements_[1]);
		double py = meas_package.raw_measurements_[0]*std::sin(meas_package.raw_measurements_[1]);
		
		x_ << px, py, 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
		/**
		 Initialize state.
		*/
		x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
	}

    // done initializing, no need to predict or update
    is_initialized_ = true;
	last_meas_ = meas_package;
}


inline void UKF::NormalizeAngle(double &angle)
{
	while (angle > M_PI) angle-=2.*M_PI;
	while (angle <-M_PI) angle+=2.*M_PI;
}


void UKF::GenerateAugmentedSigmaPoints(MatrixXd& Xsig_aug)
{
	// augmented mean vector
	VectorXd x_aug = VectorXd(7);
	// augmented state covariance
	MatrixXd P_aug = MatrixXd(7, 7);

	// sigma point matrix
	Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
	
	//create augmented mean state
	x_aug.head(5) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;
	
	// create augmented covariance matrix
	//std::cout << "P_ is " << std::endl << P_ << std::endl;
	P_aug.fill(0.0);
	P_aug.topLeftCorner(P_.rows(), P_.cols()) = P_;
	P_aug(5, 5) = std_a_*std_a_;
	P_aug(6, 6) = std_yawdd_*std_yawdd_;
	//std::cout << "P_aug is " << std:: endl << P_aug << std::endl;
  
	// create square root matrix
	MatrixXd A = P_aug.llt().matrixL();
	if (P_aug.llt().info() == Eigen::NumericalIssue) {
		std::cout << "llt failed! we have numerical issue" << std::endl;
		exit(-1);
		//throw std::range_error("llt numerical issue!");
	}
	A = std::sqrt(lambda_ + n_aug_)*A;
	//std::cout << "A is: " << std::endl << A << std::endl;
	
	// create augmented sigma points
	Xsig_aug.col(0) = x_aug;
	for(int i = 1; i <= n_aug_; i++) {
		Xsig_aug.col(i) = x_aug + A.col(i-1);
		Xsig_aug.col(i+n_aug_) = x_aug - A.col(i-1);
	}
}


void UKF::PredictSigmaPoints(const MatrixXd& Xsig_aug, double delta_t)
{
	int n_aug = n_x_ +2;
	
	Xsig_pred_ = MatrixXd(n_x_, 2*n_aug+1);
	
	//predict sigma points
	for (int i = 0; i< 2*n_aug+1; i++) {
		//extract values for better readability
		double p_x  = Xsig_aug(0,i);
		double p_y  = Xsig_aug(1,i);
		double v    = Xsig_aug(2,i);
		double yaw  = Xsig_aug(3,i);
		double yawd = Xsig_aug(4,i);
		
		double nu_a = Xsig_aug(5,i);
		double nu_yawdd = Xsig_aug(6,i);

		//predicted state values
		double px_p, py_p;

		//avoid division by zero
		if (fabs(yawd) > 0.0001) {
			px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
			py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
		}
		else {
			px_p = p_x + v*delta_t*cos(yaw);
			py_p = p_y + v*delta_t*sin(yaw);
		}
		
		double v_p = v;
		double yaw_p = yaw + yawd*delta_t;
		double yawd_p = yawd;
		
		//add noise
		px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
		py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
		v_p = v_p + nu_a*delta_t;

		yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
		yawd_p = yawd_p + nu_yawdd*delta_t;

		// normalize yaw
		NormalizeAngle(yaw_p);

		//write predicted sigma point into right column
		Xsig_pred_(0,i) = px_p;
		Xsig_pred_(1,i) = py_p;
		Xsig_pred_(2,i) = v_p;
		Xsig_pred_(3,i) = yaw_p;
		Xsig_pred_(4,i) = yawd_p;
	}
}


/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	/**
	Complete this function! Make sure you switch between lidar and radar
	measurements.
	*/
	
	// initialize state vector if not yet
	if(!is_initialized_) {
		InitMeasurement(meas_package);
		return;
	}

	/* prediction */
	double delta_t = (double)(meas_package.timestamp_ - time_us_)/1000000.0;
	Prediction(delta_t);
	
	/* then update base on measurement */
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
		UpdateRadar(meas_package);
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
		UpdateLidar(meas_package);
	}
	last_meas_ = meas_package;
	time_us_   = meas_package.timestamp_;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) 
{
	/**
	Complete this function! Estimate the object's location. Modify the state
	vector, x_. Predict sigma points, the state, and the state covariance matrix.
	*/
	// calculate predicted sigma points
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_ +1);
	GenerateAugmentedSigmaPoints(Xsig_aug);
	PredictSigmaPoints(Xsig_aug, delta_t);
	
	x_ = Xsig_pred_*weights_m_;
	P_.fill(0.0);
	for(int i = 1; i < 2*n_aug_ +1; i++) {
		//VectorXd x_diff = Xsig_pred_.col(i) - Xsig_pred_.col(0);
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		NormalizeAngle(x_diff(3));
		P_ = P_ + weights_c_(i)*x_diff*x_diff.transpose();
	}
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	/**
	Complete this function! Use lidar data to update the belief about the object's
	position. Modify the state vector, x_, and covariance, P_.
	
	You'll also need to calculate the lidar NIS.
	*/
	int n_z = 2;
	
	MatrixXd H = MatrixXd(n_z, n_x_);
	H << 1, 0, 0, 0, 0,
		 0, 1, 0, 0, 0;
	
	MatrixXd R = MatrixXd(n_z, n_z);
	R << std_laspx_*std_laspx_, 0,
		 0, std_laspy_*std_laspy_;
		  
	VectorXd y  = meas_package.raw_measurements_ - H*x_;
	MatrixXd Ht = H.transpose();

	MatrixXd S = H*P_*Ht + R;
	MatrixXd K = P_*Ht*S.inverse();
	MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
  
	// measurement update
	x_ = x_ + K*y;
	P_ = (I - K*H)*P_;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	/**
	Complete this function! Use radar data to update the belief about the object's
	position. Modify the state vector, x_, and covariance, P_.

	You'll also need to calculate the radar NIS.
	*/
	
	MatrixXd	Zsig_pred;  // sigma points in rada measurement space
	VectorXd	z_pred, z;     // mean sigma points
	MatrixXd	S;          // measurement covariance
	int			n_z = 3;
	
	if(meas_package.sensor_type_ != MeasurementPackage::RADAR) {
		cout << "UKFRadarMeasurement::Init error invalid sensor_type! " 
			 << meas_package.sensor_type_ << endl;
		return;
	}
	z = meas_package.raw_measurements_;
		
	Zsig_pred = MatrixXd(n_z, 2 * n_aug_ + 1);
	z_pred    = VectorXd(n_z);
	
	//transform sigma points into radar measurement space
	for(int i = 0; i < 2*n_aug_ +1; i++) {
		double px  = Xsig_pred_(0, i);
		double py  = Xsig_pred_(1, i);
		double v   = Xsig_pred_(2, i);
		double phi = Xsig_pred_(3, i);
		
		if(px < 0.000001) px = 0.00001;
		Zsig_pred(0, i) = std::sqrt(px*px + py*py);
		if(std::fabs(Zsig_pred(0, i)) < 0.00001) {
			std::cout << "rho is zero, invalid px/py!" << std::endl;
			return;
		}
		Zsig_pred(1, i) = std::atan2(py, px);
		Zsig_pred(2, i) = (px*std::cos(phi)*v + py*std::sin(phi)*v)/Zsig_pred(0, i);
	}
	
	//calculate mean predicted measurement
	z_pred = Zsig_pred*weights_m_;
	// normalize phi
	NormalizeAngle(z_pred(1));
	
	//calculate measurement covariance matrix S
	S = MatrixXd(n_z, n_z);
	S.fill(0.0);
	for(int i = 1; i < 2*n_aug_ +1; i++) {
		//VectorXd z_diff = Zsig_pred_.col(i) - Zsig_pred_.col(0);
		VectorXd z_diff = Zsig_pred.col(i) - z_pred;
		// angle normalization
		NormalizeAngle(z_diff(1));
		S = S + weights_c_(i)*z_diff*z_diff.transpose();
	}
  	
	// add R noise covariance
	S(0, 0) += std_radr_*std_radr_;
	S(1, 1) += std_radphi_*std_radphi_;
	S(2, 2) += std_radrd_*std_radrd_;
	
	MatrixXd Tc = MatrixXd(n_x_, n_z);
	
	Tc.fill(0.0);
	//calculate cross correlation matrix
	for(int i = 1; i < 2 * n_aug_ + 1; i++) {
		//VectorXd Xsig_diff = Xsig_pred_.col(i) - Xsig_pred_.col(0);
		VectorXd Xsig_diff = Xsig_pred_.col(i) - x_;
		NormalizeAngle(Xsig_diff(3));
		
		VectorXd z_diff = Zsig_pred.col(i) - z_pred;
		NormalizeAngle(z_diff(1));
		
		Tc = Tc + weights_c_(i)*Xsig_diff*z_diff.transpose();
	}
	
	//calculate Kalman gain K;
	MatrixXd K(n_x_, n_z);
	K = Tc*S.inverse();
	
	//update state mean and covariance matrix
	x_ += K*(z - z_pred);
	P_ -= K*S*K.transpose();
}



















