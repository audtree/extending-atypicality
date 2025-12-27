from numpy import np
from src.atypicality import knn_score, kde_score, joint_mvn_score, log_joint_mvn_score, log_joint_mvn_score, lognormal_score, gmm_score, jointlognormal_score, jointskewt_score

def test_scores():
    """
    Test the knn_score, kde_score, and joint_mvn_atypicality_score, log_joint_mvn_score functions
    using synthetic data.
    """

    # Define synthetic dataset
    dataset = [
        (np.array([1.0, 2.0]), 3.5),
        (np.array([3.0, 1.0]), 2.5),
        (np.array([0.5, 1.5]), 3.7),
        (np.array([1.5, 2.5]), 3.2),
        (np.array([2.0, 3.0]), 4.0)
    ]

    # Define input point
    input_point = (np.array([1.0, 2.0]), 3.4)

    # Test knn_score
    try:
        k = 3  # Number of neighbors for KNN
        knn_result = knn_score(input_point, dataset, k)
        assert isinstance(knn_result, float), "KNN score should return a float."
        print(f"KNN score test passed. Result: {knn_result}")
    except Exception as e:
        print(f"KNN score test failed. Error: {e}")

    # Test kde_score
    try:
        kde_bandwidth = 0.5  # Bandwidth for KDE
        kde_result = kde_score(input_point, dataset, gaussian_kernel)
        assert isinstance(kde_result, float), "KDE score should return a float."
        print(f"KDE score test passed. Result: {kde_result}")
    except Exception as e:
        print(f"KDE score test failed. Error: {e}")

    # Test joint_mvn_atypicality_score
    try:
        log_joint_mvn_result = log_joint_mvn_score(input_point, dataset)
        assert isinstance(log_joint_mvn_result, float), "Joint MVN score should return a float."
        print(f"Log Joint MVN score test passed. Result: {log_joint_mvn_result}")
    except Exception as e:
        print(f"Log Joint MVN score test failed. Error: {e}")

    try:
        lognormal_result = lognormal_score(input_point, dataset)
        assert isinstance(lognormal_result, float), "Log Normal score should return a float."
        print(f"Log Normal score test passed. Result: {lognormal_result}")
    except Exception as e:
        print(f"Log Normal score test failed. Error: {e}")

    try:
        gmm_result = gmm_score(input_point, dataset)
        assert isinstance(gmm_result, float), "GMM score should return a float."
        print(f"GMM score test passed. Result: {gmm_result}")
    except Exception as e:
        print(f"GMM score test failed. Error: {e}")

    ### Other atypicality scores not used in the proof of concept experiment
    # Test joint_mvn_atypicality_score
    try:
        joint_mvn_result = joint_mvn_score(input_point, dataset)
        assert isinstance(joint_mvn_result, float), "Joint MVN score should return a float."
        print(f"Joint MVN score test passed. Result: {joint_mvn_result}")
    except Exception as e:
        print(f"Joint MVN score test failed. Error: {e}")

    try:
        jointlognormal_result = jointlognormal_score(input_point, dataset)
        assert isinstance(jointlognormal_result, float), "Joint Lognormal score should return a float."
        print(f"Joint Lognormal score test passed. Result: {jointlognormal_result}")
    except Exception as e:
        print(f"Joint Lognormal score test failed. Error: {e}")
    try:
        jointskewt_result = jointskewt_score(input_point, dataset)
        assert isinstance(jointskewt_result, float), "Joint Skew T score should return a float."
        print(f"Joint Skew T score test passed. Result: {jointskewt_result}")
    except Exception as e:
        print(f"Joint Skew T score test failed. Error: {e}")
