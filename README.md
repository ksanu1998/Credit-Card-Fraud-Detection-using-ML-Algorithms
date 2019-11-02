# Credit Card Fraud Detection using ML Algorithms

This is an ongoing course project for EE 401 Pattern Recognition and Machine Learning, being done in collaboration with three of my classmates.
<br>
<br>
<b>
  About
</b>
<p> 
Credit card fraud can be defined as, <i> ‘Unauthorized account activity involving a payment card, by a person for which the         account is not intended’</i>. These frauds cost consumers and banks millions of dollars worldwide, as a response to which several modern fraud-detection techniques are in place today.
</p>
<b>
Data Resources
</b>
<p> 
We are working on the dataset provided in <a href="https://www.kaggle.com/mlg-ulb/creditcardfraud">Kaggle</a>, which is available under Open Database license.
</p>
<b>
  Proposed Algorithm
</b>
<p>
We decompose the proposed algorithm into three parts:<br>
• Bagging of feature vectors <br>
   Random selections tend to retain data proportions, hence the constituent imbalance levels in training data are carried         forward to the base learners. This leads to data imbalance affecting the training process to a large extent. The proposed model enables balanced data selection such that the effects of data imbalance are considerably reduced during model training.<br>
• Assigning weights to false positives and false negatives in the cost function <br>
  In our proposed approach, we may penalize mistakes made in classifying positive classes more than the mistakes made in classifying negative class or vice-versa.
  <br>
• Using SVM model with Gaussian kernel for classification
</p>
