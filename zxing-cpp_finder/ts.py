import pybinarize

img = imageio.imread("test.png")
bin_img = pybinarize.binarize(img)
imageio.imwrite("binarized.png", bin_img)
