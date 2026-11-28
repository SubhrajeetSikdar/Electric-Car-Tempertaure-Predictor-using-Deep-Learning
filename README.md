# Electric Motor Temperature Prediction Using Deep Learning

This project uses deep learning techniques with PyTorch to predict electric motor temperature based on operational and sensor data. A Convolutional Neural Network (CNN) is implemented to analyze sequential motor measurements and predict Permanent Magnet (PM) temperature.

## Project Overview


Electric motor temperature prediction is important for monitoring motor performance, preventing overheating, and supporting predictive maintenance in electric vehicles. This project applies a CNN-based regression model to predict motor temperature using selected operational parameters.

## Dataset

The dataset used in this project is the **Electric Motor Temperature Dataset**, available on Kaggle.

- **Dataset:** Electric Motor Temperature Data
- **Source:** [Kaggle - Electric Motor Temperature](https://www.kaggle.com/wkirgsn/electric-motor-temperature)
- **Dataset Size:** 998,070 records
- **Format:** CSV

### Dataset Features

The dataset includes motor operating parameters such as:

- Ambient temperature
- Coolant temperature
- Motor speed
- Torque
- Voltage components (u_d, u_q)
- Current components (i_d, i_q)
- Permanent magnet temperature (pm)
- Stator yoke temperature
- Stator tooth temperature
- Stator winding temperature
- Profile ID

## Model Architecture

The project uses a Convolutional Neural Network (CNN) built using PyTorch.

The model consists of:

- **Convolutional layers:** Extract patterns from sequential motor sensor data.
- **ReLU activation:** Introduce non-linearity.
- **Linear layers:** Perform regression and generate temperature predictions.
- **Output layer:** Predict the target motor temperature.

## Data Preprocessing

- Selected motor operational parameters as input features.
- Filtered data using a specific profile ID.
- Applied Min-Max Scaling for feature normalization.
- Converted data into sequential tensors.
- Prepared training and testing datasets.

## Training

The model is trained using:

- **Framework:** PyTorch
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Learning Rate:** 0.001
- **Epochs:** 50
- **Sequence Length:** 5
- **Test Size:** 5%

The training process minimizes prediction error by updating the neural network parameters across multiple epochs.

## Evaluation

The model performance is evaluated using:

- **R² Score:** Measures the model's predictive performance.
- **Loss Visualization:** Compares training and testing loss.
- **Predicted vs Actual Plot:** Visualizes the relationship between predicted and actual temperature values.

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn torch scikit-learn
```

## Usage

1. Download the dataset from Kaggle.
2. Place the CSV file in the appropriate dataset directory.
3. Open the `cnn-motortemperature-regression.ipynb` notebook.
4. Run the notebook cells sequentially to preprocess the data, train the model, and evaluate the predictions.

## Project File

- `cnn-motortemperature-regression.ipynb` – Data preprocessing, CNN model implementation, training, evaluation, and visualization.

## Acknowledgments

- Dataset: [Electric Motor Temperature Dataset - Kaggle](https://www.kaggle.com/wkirgsn/electric-motor-temperature)
- PyTorch: https://pytorch.org/
- Scikit-learn: https://scikit-learn.org/

