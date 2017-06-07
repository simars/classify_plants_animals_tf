<hr>

<hr>
# Classifying Plant types using Transfer Learning
# References
  
  [IJARCET-VOL-5-ISSUE-11-2664-2669.pdf](http://ijarcet.org/wp-content/uploads/IJARCET-VOL-5-ISSUE-11-2664-2669.pdf)
 
  [CodeLab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/?utm_campaign=chrome_series_machinelearning_063016&utm_source=gdev&utm_medium=yt-desc#0) by Google as a guide. Also this [tutorial](https://www.tensorflow.org/versions/r0.9/how_tos/image_retraining/index.html) is quite helpful.

  [DataSet](http://www.plant-phenotyping.org/CVPPP2014-dataset).
 

##Requirements
s
* Java 1.7+, maven

##Usage 

1. Checkout / Import Maven this maven project

2. Run the LabelImageH3 (for plants) script to label the image. `mvn -e compile exec:java -Dexec.mainClass="LabelImageH3"  -Dexec.args="inception_plants3h <path_to_file>"`

3. Run the LabelImage (for animals) script to label the image. `mvn -e compile exec:java -Dexec.mainClass="LabelImage"  -Dexec.args="inception_animals5h <path_to_file>"`

<hr>

