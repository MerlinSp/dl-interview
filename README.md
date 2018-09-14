# dl-interview

# How to run

There is only one script in the repo that produces all the desired results. It trains a net with the training data that was provided, saves the weights, saves the submission csv file with the predictions on the test images, and finally, saves the test images with corrected rotation to a folder.

To use simply run 'python <training images path> <training data ground truth csv> <test images path>'

Example: 

You can find the submission csv and the zip with the test images with corrected rotations in the parent folder of this repository.


# Approach

I tried using the first simple approach of a CNN that I found online. Mnist didn't seem apprapriated and maybe too simple for the problem that I wanted to solve. As the Cifar architecture already archived good results in object classification on images, I decided to give it a try. I split 10% of the training data for validation, to make sure the net generalizes best possible to data it has never seen. I impleented a Earlystoping monnitor, and a model checkpoint that should save the best weights based on the validation accuracy. The best training result on val_acc ended up being 0.9902.
