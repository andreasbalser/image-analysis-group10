User your two final image classifiers (one for letters, and one for digits) to classify all of our test images. Report the predicted classes for all of your self-generated images. Are the results satisfying?

Conclude your project documentation with a short discussion of quality of the achieved classification results. Did you face any significant difficulties? Do you see need for further improvement of the image classifiers? Have you noticed any clear signs of overfitting?

# Final Classification Results and Discussion

## Classification Results

We used 50 self-generated test images (29 letters and 21 digits).

- digit classifier:
  - Several misclassifications occurred, particularly with visually similar digits

-  letter classifier
  - Performed poorly, the majority of predictions were incorrect or seemed random.

### Are the results satisfying?

Overall, the results are not satisfying. While the digit classifier showed some ability to generalize, both models struggled withself-generated handwriting samples, despite high validation accuracy on EMNIST.

## Discussion

  The EMNIST dataset contains centered, clean, and standardized characters. In contrast, our own handwritten samples vary in style, stroke thickness, alignment, and shape.

  There are clear signs of overfitting — particularly for the letter classifier. The models perform very well on validation data.

## Need for Further Improvements

Yes — to achieve reliable results, the classifiers need several improvements, like probably a larger data-set (or also different ones so the models are not trained on a very specific EMNIST style), or data augmentation
