# Amazon ML Challenge --- Model Documentation

### **1. Methodology**

The solution follows a **transfer learning** approach using a
pre-trained convolutional neural network to leverage large-scale image
representations and adapt them to the given task. The dataset consists
of product images and accompanying metadata. The goal was to predict
product attributes (e.g., price or category) accurately by combining
visual features with simple preprocessing techniques.

Data preprocessing included reading and normalizing all images to a
consistent size of **224 × 224 px** and ensuring data cleanliness by
validating image files. K-fold cross-validation (K = 3) was used to
improve generalization and reduce overfitting. Model training was
monitored using early stopping and dynamic learning-rate reduction.

------------------------------------------------------------------------

### **2. Model Architecture / Algorithms**

-   **Base model:** Pre-trained **ResNet50** (from
    `tensorflow.keras.applications`), with ImageNet weights and the top
    classification layers removed.\
-   **Custom head:**
    -   Global Average Pooling layer\
    -   Fully connected dense layers with ReLU activation\
    -   Dropout and L2 regularization for overfitting control\
    -   Output layer (softmax for classification / linear for
        regression)\
-   **Training setup:**
    -   Optimizer: Adam\
    -   Loss: categorical cross-entropy or MSE (depending on target
        type)\
    -   Batch size: 32 • Epochs: 3 • Image size: 224 × 224\
    -   Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

This architecture balances computational efficiency with strong feature
extraction performance, taking advantage of deep residual learning for
robust image representations.

------------------------------------------------------------------------

### **3. Feature Engineering**

-   **Image Preprocessing:** Normalization of pixel values (rescaling =
    1/255).\
-   **Augmentation:** Random flips, rotations, and zooms via
    `ImageDataGenerator` to enhance data diversity.\
-   **Metadata (if present):** Scaled numeric features using
    `MinMaxScaler` before concatenation.\
-   **Cross-validation:** 3-fold K-Fold strategy for stable
    out-of-sample estimates.

------------------------------------------------------------------------

### **4. Additional Information**

-   **Frameworks:** TensorFlow 2 / Keras + Scikit-Learn + Pandas + PIL\
-   **Hardware:** Kaggle GPU runtime (Tesla T4).\
-   **Evaluation metric:** RMSE / Accuracy depending on final task
    formulation.\
-   **Reproducibility:** All random seeds fixed; training scripts use
    deterministic image pipelines.
