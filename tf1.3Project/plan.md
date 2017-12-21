1. visualize learning

    finished: in windows, we can visualize it, even the inputdata mudual, has the tensors to visualize;

1. save the trained parameter

    reference: the train.py has save the parameter, and the freeze.py load the model and parameters;
    modify the 0.2TrainableParameters.py 
        <-- do the same experiment as the tutorial
            https://www.tensorflow.org/api_docs/python/tf/train/Saver
            https://www.tensorflow.org/programmers_guide/saved_model


## important record
savepara_file = 'model.ckpt' #NameError: name 'FLAGS' is not defined
savepara_dir = './save_parameter/'    
saver.save(sess,savepara_dir+savepara_file)
then I will get 4 files

checkpoint
savepara_file.data-00000-of-00001
savepara_file.index
savepara_file.meta

suffix  .data-00000-of-00001  .index  .meta


saver.restore(sess, savepara_dir + savepara_file)
print("W,b value after restore: ", sess.run([W, b]))

get the right thing !!

then will the global step


if I add the global step saver.save(sess, savepara_dir + savepara_file, global_step=i)
the savepara_file get a suffix of -i


1. save and check the input file, don't download every time

    reference: train.py download from url