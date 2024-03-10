# computer-vision
I love you homies <3

Some library functions aren't allowed, but we could use them temporarily to quickly get a working solution then retroactively replace them with our own implementations.
This would also give us some explar results to regress on later down the line. 

## Running locally
1. Install python etc
2. (Optional) setup venv in '.venv' folder so that jupyter knows it exists
```shell
python3 -m venv .venv
source bin/activate
```

3. Install dependencies
```shell
pip install numpy pandas opencv-python
```

To use the jupyter notbook you'll need some more
```shell
pip install numpy pandas opencv-python ipykernel matplotlib
```

4. Run the specified task by providive the correct dataset. E.g. to run task 1
```shell
python main.py --Task1Dataset ./Task1Dataset
```

## Task 1 
Calculate the (smaller) angle between two lines in a black and white image.

For example, for the image below it should return 40.

![Two lines at a 40 degree angle](Task1Dataset/image1.png "Two lines at a 40 degree angle")