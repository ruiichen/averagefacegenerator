# Average Face Generator

A fun tool to generate the average face out of many pictures utilizing machine learning.

Check it out [here](http://r9chen.pythonanywhere.com/)

## 💡Inspiration

One of the most interesting studies that I've read is the phenomenon of averageness, where the composite face generated with a multitude of averaging techniques is consistently more attractive than that of those faces used to generate it. When I learned about the emergence of machine learning, this was one of the first things that came to mind.

## 🔍 What it does

The average face generator is a website that takes pictures of faces that the user submits and generates the average face. [Try it](http://r9chen.pythonanywhere.com/) by submitting all the images of faces you want to combine at once, and after a brief period, the composite face will show up on screen.

![image](https://github.com/ruiichen/averagefacegenerator/assets/114363176/c42a0107-35d9-4ef9-b875-c6a421cd5313)

## ⚙️ How it was built

The first thing that was built was the algorithm to generate the faces.
<ul>
  <li> From each facial image uploaded, dlib was used to calculate 68 facial landmarks.</li>
  <li> The images were then normalized to set dimensions using a similarity transform and overlapped by the corners of their eyes. </li>
  <li> The average of each transformed landmark point was calculated. </li>
  <li> A Delaunay Triangulation was calculated using the averaged facial landmarks. </li>
  <li> The Delaunay Triangulation is used against each image to warp each face to the shape of the average face. </li>
</ul>

## 🚧 Challenges I ran into

The main challenge of the project was the handling of the user's uploaded images. The algorithm was written with local files in mind, so I had to learn how to convert uploaded images into something the algorithm could use.

## 📚 What I learned

<ul>
  <li>Learned how to integrate the algorithm into a website.</li>
  <li>Learned how to deploy the final product.</li>
</ul>
