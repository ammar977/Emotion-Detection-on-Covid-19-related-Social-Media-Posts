This directory contains the files used for the bonus (East-Asian face recognition) section of the project.
- Bonus_Face_Recognition contains the working Jupyter Notebook.
- Bonus_Face_Recognition_scc contains the version of above that ran on the SCC, with the data augmentation code removed. I'm keeping it just in case I missed a change from the above.
- face_recognition_2 contains my scratch work and the proof-of-concept version of the code found in the proper one. I'm keeping it just in case the file manipulation in the proper one doesn't work.
- Model_results contains the 5-fold cross-validation information for the various models tried.

The code will require images.zip to be unzipped into "../Data Instagram/" to produce the correct file. This is a one-time requirement, once you have "images.tfrecord" you won't need any other data file.

On the _scc file, I made multiple changes to the function build_model3(). These models are all listed in Bonus_Face_Recognition as build_model4 and build_model5. These will change the weighting from the original ones automatically. To change the number of epochs, change the number directly in fit() in the cross-validation loop.