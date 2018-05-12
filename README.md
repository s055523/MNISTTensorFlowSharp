# MNISTTensorFlowSharp
使用TensorFlowSharp训练MNIST数据集

http://www.cnblogs.com/haoyifei/p/8654743.html

TensorflowSharp的源码地址：https://github.com/migueldeicaza/TensorFlowSharp

如果在运行时发现问题“找不到libtensorflow.dll”，则需要访问

http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow-windows/lastSuccessfulBuild/artifact/lib_package/libtensorflow-cpu-windows-x86_64.zip

下载这个压缩包。然后，在下载的压缩包中的\lib中找到tensorflow.dll，将它改名为libtensorflow.dll，并在你的工程中引用它。
