User your two final image classifiers (one for letters, and one for digits) to classify all of our test images. Report the predicted classes for all of your self-generated images. Are the results satisfying?

Conclude your project documentation with a short discussion of quality of the achieved classification results. Did you face any significant difficulties? Do you see need for further improvement of the image classifiers? Have you noticed any clear signs of overfitting?

# Final Classification Results and Discussion

## Classification Results

We used our two final image classifiers — one for **letters** and one for **digits** — to classify 50 self-generated test images (29 letters and 21 digits).

- **Digit classifier** (trained via transfer learning on EMNIST Digits):
  - ✅ Achieved reasonable performance with **~9 out of 21** digits correctly classified.
  - ❌ However, several misclassifications occurred, particularly with visually similar digits (e.g., 2 vs. 3, 6 vs. 0).

- **Letter classifier** (trained on EMNIST Letters):
  - ❌ Performed poorly with only **~5 out of 29** letters correctly classified.
  - ❌ The majority of predictions were incorrect or seemed random.

### Are the results satisfying?

Overall, the results are **not satisfying** — particularly for letter classification. While the digit classifier showed some ability to generalize, both models struggled significantly with real, self-generated handwriting samples, despite high validation accuracy on EMNIST.

---

## Discussion

### Difficulties Faced

- **Domain Shift**:  
  The EMNIST dataset contains centered, clean, and standardized characters. In contrast, our own handwritten samples vary in style, stroke thickness, alignment, and shape, leading to a domain mismatch.

- **Preprocessing**:  
  Despite applying autocontrast, padding, and resizing to 28×28, the resulting images still differed significantly from EMNIST samples. Small visual mismatches likely led to many misclassifications.

- **Overfitting**:  
  There are clear signs of **overfitting** — particularly for the letter classifier. The models perform very well on validation data but poorly on unseen handwriting, indicating poor generalization.

---

## Need for Further Improvements

Yes — to achieve reliable results, the classifiers need several improvements:

1. **Data Augmentation**:  
   Apply transformations such as rotation, scaling, translation, and noise to increase variability and robustn
