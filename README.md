# error-frames-calculator
An implementation of [motion compensation](https://en.wikipedia.org/wiki/Motion_compensation), used in video compression.

## Description
* The program takes a video as an input (make sure video's filename is correct and in the same folder as the executable).
* Promts the user to select either whole or block-based frame prediction.
* Calculates the error frames. For block-based frame prediction, Two Dimensional Logarithmic Search (TDLS) is used.
* Returns a video with the error frames.

(Note: Installation of Python libraries cv2 and numpy is required.)

## Video examples
1. Input video of 60 frames:  
![](/examples/gifs/exm.gif "Input video")

2. Output videos:
* Whole frame difference  
![](/examples/gifs/wh_err_exm.gif "Whole frame difference")

* Block-based difference (Algorithm indulgence: lenient/4)  
![](/examples/gifs/b_b_err__0.1__exm.gif "Block-based difference (Algorithm indulgence: lenient)")

* Block-based difference (Algorithm indulgence: strict/0)  
![](/examples/gifs/b_b_err__0__exm.gif "Block-based difference (Algorithm indulgence: strict)")
