# DualReconstruction

This reposity is organized mainly for an image decomposition algorithm which is proposed to solve the image reconstruction and image-domain decomposition problem in Dual-energy Computed Tomography (DECT). <br>

The algorithm is designed based on deep learning paradigm. For more theoretical details, please go to [Deep Learning](http://www.deeplearningbook.org/) and [Material Decomposition Using DECT](https://pubs.rsna.org/doi/10.1148/rg.2016150220).<br>

The code is currently based on python 3.6, [Tensorflow](https://github.com/tensorflow/tensorflow) 1.4.0 and [ODL](https://github.com/odlgroup/odl) in Windows 7 platform. <br>

  * data: contains 3 paths.
    * system_matrix: the system matrix used in the reconstruction algorithms. You can generate the matrix by running the 'iterative_reconstruction.m' or 'FBP_reconstruction.m' file in the src path.
    * testing_set: The data used for testing.
    * training_set: The data used for training the deep model. Both training and testing set can be download from [here](https://pan.baidu.com/s/1VfhTuNenuy2C6HAw1aWbZA)(Extraction number: t4ya).<br>
  * log: save the Tensorflow log file in training process.
  * model: save the trained model
  * result: save the result generated by the reconstruction algorithms.
  * src: the codes for the proposed algorithm and two other competing ones:
    * Filter back projection (FBP) followed by direct matrix inversion (FBP_reconstruction.m)
    * Combined iterative reconstruction and image decomposition (iterative_reconstruction.m). Related paper: [Combined iterative reconstruction and image-domain decomposition for dual energy CT using total-variation regularization.](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1118/1.4870375)
    * The proposed deep model (main.py). You can start to train the proposed deep model via the cmd:
    ```bash
    python main.py --dataset="../data/training_set/" --mode="train" --model_name="your-saved-model-result-name" --lr = 0.0001 --epoch=30 --model_step=1000 --batch_size=1
    ```
      After finishing the training process, you can test the trained model via the cmd:
     ```bash
    python main.py --dataset="../data/testing_set/" --mode="feedforward" --model_name="your-saved-model-result-name" --checkpoint="../model/your-saved-model-result-name"
    ```
    
# Contact
 Email: vastcyclone@yeah.net
