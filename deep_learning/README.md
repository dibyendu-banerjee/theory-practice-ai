<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Deep Learning Modules</title>
  <style>
    body {
      font-family: 'Segoe UI', sans-serif;
      line-height: 1.6;
      background-color: #fdfdfd;
      color: #333;
      padding: 2rem;
    }
    h1, h2 {
      color: #2c7be5;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 1rem 0;
    }
    th, td {
      border: 1px solid #ccc;
      padding: 0.75rem;
      text-align: left;
    }
    th {
      background-color: #f0f4f8;
    }
    tr:nth-child(even) {
      background-color: #f9f9f9;
    }
    code {
      background-color: #f4f4f4;
      padding: 0.2rem 0.4rem;
      border-radius: 4px;
      font-family: monospace;
    }
    .section {
      margin-bottom: 2rem;
    }
  </style>
</head>
<body>

  <h1>🌟 Deep Learning Modules</h1>
  <p><em>A curated collection of hands-on deep learning scripts covering neural networks, CNNs, transfer learning, reinforcement learning, and model optimization.</em></p>

  <div class="section">
    <h2>📁 Project Structure</h2>
    <table>
      <thead>
        <tr>
          <th>📂 Folder/File</th>
          <th>📄 Description</th>
        </tr>
      </thead>
      <tbody>
        <tr><td><code>src/</code></td><td>Source folder containing all Python modules</td></tr>
        <tr><td><code>ann_spam_classifier.py</code></td><td>ANN for spam detection + artificial neuron demo</td></tr>
        <tr><td><code>ann_loan_approval.py</code></td><td>ANN for predicting loan approval</td></tr>
        <tr><td><code>cnn_image_classification.py</code></td><td>CNN from scratch on CIFAR-10</td></tr>
        <tr><td><code>cnn_transfer_learning.py</code></td><td>Transfer learning using MobileNetV2</td></tr>
        <tr><td><code>cnn_feature_visualization.py</code></td><td>Visualize CNN feature maps</td></tr>
        <tr><td><code>dqn_agent.py</code></td><td>Q-learning and Deep Q-Network (DQN) agent</td></tr>
        <tr><td><code>model_tuning_metrics.py</code></td><td>ROC-AUC, MSE, MAE for classification and regression</td></tr>
        <tr><td><code>model_regularization.py</code></td><td>L2 and Dropout regularization techniques</td></tr>
        <tr><td><code>model_cross_validation.py</code></td><td>K-fold cross-validation with Random Forest</td></tr>
        <tr><td><code>model_transfer_learning.py</code></td><td>Transfer learning with ResNet50</td></tr>
        <tr><td><code>knowledge_distillation.py</code></td><td>Knowledge distillation: teacher-student setup</td></tr>
        <tr><td><code>requirements.txt</code></td><td>Python dependencies for all modules</td></tr>
        <tr><td><code>README.md</code></td><td>Project overview and documentation</td></tr>
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>🧠 Module Overview</h2>
    <ul>
      <li><strong>ann_spam_classifier.py</strong> — Basic ANN for spam detection with a conceptual neuron demo.</li>
      <li><strong>ann_loan_approval.py</strong> — Predict loan approval using Keras-based ANN.</li>
      <li><strong>cnn_image_classification.py</strong> — CNN from scratch trained on CIFAR-10.</li>
      <li><strong>cnn_transfer_learning.py</strong> — Fine-tune MobileNetV2 for image classification.</li>
      <li><strong>cnn_feature_visualization.py</strong> — Visualize feature maps from CNN layers.</li>
      <li><strong>dqn_agent.py</strong> — Q-learning and Deep Q-Network (DQN) for reinforcement learning.</li>
      <li><strong>model_tuning_metrics.py</strong> — ROC-AUC, MSE, and MAE metrics with visualizations.</li>
      <li><strong>model_regularization.py</strong> — L2 and Dropout regularization to reduce overfitting.</li>
      <li><strong>model_cross_validation.py</strong> — K-fold cross-validation using Random Forest.</li>
      <li><strong>model_transfer_learning.py</strong> — Transfer learning with ResNet50 and synthetic data.</li>
      <li><strong>knowledge_distillation.py</strong> — Train a student model using soft targets from a teacher model.</li>
    </ul>
  </div>

  <div class="section">
    <h2>📦 Dependencies</h2>
    <p>All dependencies are listed in <code>requirements.txt</code>. Core libraries include:</p>
    <ul>
      <li><code>numpy</code></li>
      <li><code>pandas</code></li>
      <li><code>matplotlib</code></li>
      <li><code>scikit-learn</code></li>
      <li><code>tensorflow</code></li>
      <li><code>keras</code></li>
    </ul>
  </div>

  <hr>

  <p><strong>Author:</strong> Dibyendu Banerjee, Sourav Kairi</p>
  <p><strong>License:</strong> For educational and research use only. © 2024 Dibyendu Banerjee. All rights reserved.</p>

</body>
</html>
