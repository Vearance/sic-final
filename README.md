# sic-final

This repository is used for Samsung Innovation Campus purposes. The MongoDB uri will be deleted after the program finishes (hopefully).

## how to use (localhost)
1. run 'src/server.py' to activate the server for the sensor's data
2. check if the post server on the arduino IDE is correct, then turn on the esp32. the response from the sensor can be checked from 'server.py' terminal
3. run 'streamlit run main.py' on '../src' root directory
4. click on the "Start real-time testing" button

### sensor models
different places come with different temperature and humidity conditions. the original "iforest_pipeline" was trained on data from the Jakarta area, while the "iforest_pipeline_bdg" was trained on data from the Bandung area.

### webcam
for this source code (main.py), I used cv2.VideoCapture(1). this means if you want to use your laptop camera, you should change 'main.py' on Line 187, Column 25.