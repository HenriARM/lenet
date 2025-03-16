# TODO

Part 4-6
Prepare results
Choice

# Theoretical
- read LeNet paper
- try Kaggle free resources and Azure (students or the one we have)
- implement custom Conv layer
- go through my Evalds CV pages to finish them
- Kaiming init https://paperswithcode.com/method/he-initialization
- try AI model explanation tools like Captum

- are the classes balanced? if not upsamplpe with generated pictures 


CHOICE 1: Create and apply a function to decrease the learning rate at a 1/2 of the value every 5 epochs: 5. Add a graph of the learning rate over time.
CHOICE 2: Instead of having a fixed validation set, implement k-fold cross-validation: 10. Note that this significantly increases the running time. Use k=5.
CHOICE 3: Do a hyperparameter search for optimizer (evaluate 3 options), learning rate (evaluate 3 options), weight decay (evaluate 2 options). If you do a grid search: max 5 points. If you use evolutionary search, max 10 points. In this case, you also need to evaluate two different batch sizes.
CHOICE 4: Create output layers at different parts of the network for additional feedback. Show and explain some outputs of a fully trained network: 15. The output layers should connect to the convolution or pooling layers.
CHOICE 5: Perform data augmentation techniques (at least 3): 5. Report and explain how they affect your performance. Only select meaningful techniques.
CHOICE 6: Provide t-SNE visualization of the fully connected layer before your output layer: 10. Show a graph with the embeddings of the test set. Discuss the graph in terms of the observed and expected confusions.
CHOICE 7: Evaluate cross-dataset performance on the Tiny ImageNet dataset (download from Huggingface or Kaggle). Pre-process the images so they fit your network, choose the classes that overlap with CIFAR-10 and report the test performance on one of your models with accuracy and confusion matrix. Discuss the differences with CIFAR-10: 15. 
CHOICE 8: Fine-tune your best CIFAR-10 model on the Tiny ImageNet overlapping classes. Report accuracy and confusion matrices. Compare model trained from scratch (CHOICE 7) with this fine-tuned model: 5 points.


Using device: cpu
Files already downloaded and verified
Files already downloaded and verified
Epoch 1/10, Train Loss: 1.6145, Train Acc: 41.38%, Val Loss: 1.4107, Val Acc: 49.55%
Epoch 2/10, Train Loss: 1.3275, Train Acc: 52.33%, Val Loss: 1.3121, Val Acc: 53.22%
Epoch 3/10, Train Loss: 1.2056, Train Acc: 57.21%, Val Loss: 1.2673, Val Acc: 55.25%
Epoch 4/10, Train Loss: 1.1127, Train Acc: 60.77%, Val Loss: 1.1867, Val Acc: 58.44%
Epoch 5/10, Train Loss: 1.0383, Train Acc: 63.18%, Val Loss: 1.1648, Val Acc: 59.22%
Epoch 6/10, Train Loss: 0.9719, Train Acc: 65.67%, Val Loss: 1.1871, Val Acc: 58.89%
Epoch 7/10, Train Loss: 0.9191, Train Acc: 67.50%, Val Loss: 1.1862, Val Acc: 59.08%
Epoch 8/10, Train Loss: 0.8712, Train Acc: 69.16%, Val Loss: 1.1987, Val Acc: 58.88%
Epoch 9/10, Train Loss: 0.8201, Train Acc: 71.11%, Val Loss: 1.2300, Val Acc: 59.02%
Epoch 10/10, Train Loss: 0.7738, Train Acc: 72.57%, Val Loss: 1.2440, Val Acc: 59.84%
Model saved to lenet5.pth
