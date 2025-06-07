![image](https://github.com/user-attachments/assets/0938b2c8-5423-4a89-9a2b-6f9c2dcb41bf)


According to the DBSCAN algorithm, a special label is assigned to each example (data point) using
the following criteria: <br>
  • A point is considered a core point if at least a specified number (MinPts) of neighboring points fall within the specified radius, 𝜀 <br>
  • A border point is a point that has fewer neighbors than MinPts within 𝜀 , but lies within the 𝜀 radius of a core point <br>
  • All other points that are neither core nor border points are considered noise points <br>
