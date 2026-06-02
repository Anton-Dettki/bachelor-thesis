# Dataset Documentation
url: https://zenodo.org/records/15712834

From the online description: 
20 Participants performing five ADL activities in an appt

## adl_noerror
1. **Make a phone call** The recorded message provides cooking directions, which the participant summarizes on a notepad
2. **Wash hands**
3. **Cook** The participant cooks a pot of oatmeal according to the directions given in the phone message
4. **Eat**
5. **Clean**

## adl_error 
1. **Make a phone call ERROR** The recorded message provides cooking directions, which the participant summarizes on a notepad
2. **Wash hands ERROR**
3. **Cook ERROR** The participant cooks a pot of oatmeal according to the directions given in the phone message
4. **Ea ERRORt**
5. **Clean ERROR**

The files are named according to the **participant number** and **task number** (e.g., p01.t1.csv contains sensor data for participant 1 performing task 1). There is **one sensor reading in each row** with fields date, time, sensor, and message.

A floorplan is provided in Chinook.png together with the location of the sensors. The sensors are categorized and named as:
- **M01 - M026**: PIR Motion Detectors (ON when detecting motion and OFF when it stops)
- **I01 - I08**: item use sensors for (in order) oatmeal, raisins, brown sugar, bowl, measuring spoon, medicine container, pot, phone book (PRESENT or ABSENT indicating item is on sensor or not)
- **D01**: Door sensor on kitchen cabinet
- **AD1-A and AD1-B**: Water sensors for kitchen sink
- **AD1-C**: burner sensor
- **asterisk**: phone use sensor