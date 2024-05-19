from mpi4py import MPI
import numpy
from PIL import Image 
import numpy as np
import threading 
import queue 
import cv2

class WorkerThread(threading.Thread): 
 
    def __init__(self):
        threading.Thread.__init__(self) 
        #self.task_queue = task_queue
        self.comm = MPI.COMM_WORLD  ## comm initialization 
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        

    def run(self):

        # Create a queue for tasks 
        task_queue2 = queue.Queue()
        counter = 0

        while True: 
        
            
############### here is my addition to the code, i first check if the thread is master or slave and then act accordingly
            
            ## message tag IDS for each send receive communication
            
            Row_Splits_ID = 1
            Image_Split_ID = 2
            Image_Width_ID = 3
            Image_Depth_ID = 4
            Final_Send_ID = 5
            operation_Type_ID = 6

            ### some necessary variables 
            start_index = 0
            SplitsRows = 0
            world_size = self.comm.Get_size() # size of processes


            if (self.rank == 0):  ## if master process

                
                
                if task_queue2.empty(): 
                    
                    Num_of_Images = input('type how many images are you going to supply?\n')

                    image_Path_list = []
                    list_of_Images = [] 

                    Num = eval(Num_of_Images)

                    for n in range(Num):
                        stringx = input("insert image number " + str(n) + "'s path here please\n")
                    #image_Path_list.append(stringx)
                        list_of_Images.append(cv2.imread(stringx, cv2.IMREAD_COLOR)) #consider using only 1 process here and multiple inside class by testing size limit using print in class


                    operation_Num = input('select operation number please [1-3] ?\n')


                    if (int(operation_Num) == 1):
                        operation = 'edge_detection'
                    elif (int(operation_Num) == 2):
                        operation = 'color_inversion'
                    elif (int(operation_Num) == 3):
                        operation = 'thresholding'


                    for h in range (Num):
                        task_queue2.put((list_of_Images.pop() , operation))
                    
                    print(task_queue2.qsize())

                
                print(task_queue2.qsize())

    
                # Get a task from the queue 
                task = task_queue2.get()
                #task_queue2.pop()
                
                image, operation = task ## getting image and operation supplied from task queue 
            
                numpydata = np.array(image) # transforming image into numpy array to be able to send it later after splitting it
                ImageHeight = numpydata.shape[0] 
                ImageWidth = numpydata.shape[1]
                ImageDepth = numpydata.shape[2]

                img1Darray = numpydata.reshape((ImageHeight * ImageWidth), ImageDepth) # flattening the image into a 2d array where width and hieght becomes one dimension and depth a dimension on its own

                for i in range(1, world_size):
                 
                    SplitsRows = ImageHeight / (world_size - 1)
                    img_split = np.split(img1Darray, (world_size - 1)) # we split image over amount of processes - 1 (cause master isnt slave)
                ## we divide the total number of rows  over the total number of processes - 1 (i.e. excluding the root process)
                ## we obtain a disection of the input into x processes, i.e. Y vertical image splits were
                ## each split is x image rows and we have Y splits based on Y processes e.g. X = 32 , Y = 16, image = 512 *512
                    
                    start_index = i - 1 
                    
                ## we scatter the data across the processes, were we send data of each split by looping
                ## for each process starting from process 1 to process size - 1
                ## we send image split hieght and the data of the splitted image to a slave process
                ## and the data were the size of the data is the image peice size 
                #comm.send(ImageHeight, source = i, Image_Height_ID)
                #comm.send(ImageWidth, source = i, Image_Width_ID)

                    self.comm.send(SplitsRows, i, Row_Splits_ID)
                    self.comm.send(ImageDepth, i, Image_Depth_ID)
                    self.comm.send(ImageWidth, i, Image_Width_ID)
                    self.comm.send(operation, i, operation_Type_ID)


                    img_split2 = np.zeros((int(SplitsRows), int(ImageDepth)), dtype= numpy.uint8)
                    img_split2 = img_split[start_index]
                    self.comm.send(img_split2, i, Image_Split_ID)

                ##  i var will represent the process number in the send parameters

            elif (self.rank > 0):  ## if not a master thread

                
            ## we receive all information about either split including the data which we store in an array

                
                SplitsRows = self.comm.recv(source = 0, tag = Row_Splits_ID) 
                ImageDepth = self.comm.recv(source = 0, tag = Image_Depth_ID)
                ImageWidth = self.comm.recv(source = 0, tag = Image_Width_ID)
                operation = self.comm.recv(source = 0, tag = operation_Type_ID)

                sub_image = self.comm.recv(source = 0, tag = Image_Split_ID)
                imgResultxvb = sub_image.reshape(int(SplitsRows),int(ImageWidth),int(ImageDepth))
                img6 = Image.fromarray(imgResultxvb) 
                img6.save("SubImage" + str(self.rank) + ".png", 'PNG') ## saving each split
                retreiveImg = "SubImage" + str(self.rank) + ".png"
                SplitResults = self.process_image(retreiveImg, operation) ## passing the sub image splits to the image processor function 
                
            ## sending each split result to master processes for later aggregation

                self.comm.send(SplitResults, 0, Final_Send_ID) ## sending the results after doing the operation to the master for aggreagation 
                

            if (self.rank == 0):  ## now final aggragation of all split results

                Aggregated_Result = np.empty((0 , int(ImageWidth))) 
                
                for i in range(1 , world_size):  ### looping through each processes results

                    Partition = self.comm.recv(source = i, tag = Final_Send_ID) ## receiving the resulted images
                    Aggregated_Result = np.concatenate((Aggregated_Result,Partition), axis = 0) ## cocatenating the results
 
                filename = "Image" + str(counter) + ".png"
                cv2.imwrite(filename, Aggregated_Result) ## saving the operation results after cocnatentation
                
                #img4 = Image.fromarray(Aggregated_Result)
                
                #img4.show() ## showing the operation results after cocnatentation
                
                counter +=1
                
                

                if task_queue2.empty(): 
                    print("I broke")
                    break
                
                print(task_queue2.qsize())


                

        MPI.Finalize()

## end of our Advanced MPI processing addition ##########################################

    def process_image(self, image, operation): 
    # Load the image 
        img = cv2.imread(image, cv2.IMREAD_COLOR) 
 
    # Perform the specified operation 
        if operation == 'edge_detection': 
        # Apply Canny edge detection 
            result = cv2.Canny(img, 100, 200) 
        elif operation == 'color_inversion': 
        # Invert colors using bitwise_not 
            result = cv2.bitwise_not(img) 
    # Add more operations as needed... 
        elif operation == 'thresholding': 
        # Convert to grayscale 
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        # Apply thresholding 
            _, result = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY) 
        else: 
        # Unsupported operation 
            result = None 
 
        return result 
 
  

    def resize_image(self, image, width=None, height=None): 
        resized_image = cv2.resize(image, (width, height)) 
        return resized_image
     
    def rotate_image(self, image, angle): 
        height, width = image.shape[:2] 
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1) 
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height)) 

        return rotated_image 
    
    def apply_blur(self, image, kernel_size): 
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0) 
        return blurred_image 


#D:/test.png
#D:/test3.png

# "D:/test2.png"


#for i in range(MPI.COMM_WORLD.Get_size() - 1):
 
t2 = WorkerThread()
t2.start()
     

print("Exit")
