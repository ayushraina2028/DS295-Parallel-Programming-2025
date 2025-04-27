import numpy as np
import pandas as pd

def generate_well_separated_gmm_data(n_samples=1000, n_components=3, n_features=2, random_state=42):
    np.random.seed(random_state)
    # Set means far apart for clear separation
    means = np.array([
        [-10, -10],
        [0, 10],
        [10, -10]
    ])
    # Use small covariances for tight clusters
    covariances = [np.eye(n_features) * 0.8 for _ in range(n_components)]
    weights = np.ones(n_components) / n_components  # Equal weights

    samples = []
    labels = []
    for _ in range(n_samples):
        component = np.random.choice(n_components, p=weights)
        sample = np.random.multivariate_normal(means[component], covariances[component])
        samples.append(sample)
        labels.append(component)

    data = np.array(samples)
    df = pd.DataFrame(data, columns=[f'feature_{i+1}' for i in range(n_features)])
    df['label'] = labels
    return df

if __name__ == "__main__":
    df = generate_well_separated_gmm_data(n_samples=1000, n_components=3, n_features=2)
    df.to_csv('gmm_data.csv', index=False)
    print("Generated well-separated GMM data and saved to gmm_data.csv")