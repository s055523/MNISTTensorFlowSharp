using System;
using System.IO;
using System.IO.Compression;
using System.Linq;

namespace MNISTTensorFlowSharp
{
    /// <summary>
    /// 图片类型
    /// </summary>
    public struct MnistImage
    {
        public int Cols, Rows;
        public byte[] Data;
        public float[] DataFloat;

        public MnistImage(int cols, int rows, byte[] data)
        {
            Cols = cols;
            Rows = rows;
            Data = data;
            DataFloat = new float[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                //数据归一化（这里将0-255除255变成了0-1之间的小数）
                //也可以归一为-0.5到0.5之间
                DataFloat[i] = Data[i] / 255f;
            }
        }
    }

    // Helper class used to load and work with the Mnist data set
    public class Mnist
    {
        public MnistImage[] TrainImages, TestImages, ValidationImages;
        public byte[] TrainLabels, TestLabels, ValidationLabels;
        public byte[,] OneHotTrainLabels, OneHotTestLabels, OneHotValidationLabels;

        //三个Reader分别从总的数据库中获得数据
        public BatchReader GetTrainReader() => new BatchReader(TrainImages, TrainLabels, OneHotTrainLabels);
        public BatchReader GetTestReader() => new BatchReader(TestImages, TestLabels, OneHotTestLabels);
        public BatchReader GetValidationReader() => new BatchReader(ValidationImages, ValidationLabels, OneHotValidationLabels);

        /// <summary>
        /// 数据的一部分，包括了所有的有用信息
        /// </summary>
        public class BatchReader
        {
            int start = 0;
            //图片库
            MnistImage[] source;
            //数字标签
            byte[] labels;
            //oneHot之后的数字标签
            byte[,] oneHotLabels;

            internal BatchReader(MnistImage[] source, byte[] labels, byte[,] oneHotLabels)
            {
                this.source = source;
                this.labels = labels;
                this.oneHotLabels = oneHotLabels;
            }

            /// <summary>
            /// 返回两个浮点二维数组（C# 7的新语法）
            /// </summary>
            /// <param name="batchSize"></param>
            /// <returns></returns>
            public (float[,], float[,]) NextBatch(int batchSize)
            {
                //一张图
                var imageData = new float[batchSize, 784];
                //标签
                var labelData = new float[batchSize, 10];

                int p = 0;
                for (int item = 0; item < batchSize; item++)
                {
                    Buffer.BlockCopy(source[start + item].DataFloat, 0, imageData, p, 784 * sizeof(float));
                    p += 784 * sizeof(float);
                    for (var j = 0; j < 10; j++)
                        labelData[item, j] = oneHotLabels[item + start, j];
                }

                start += batchSize;
                return (imageData, labelData);
            }
        }

        /// <summary>
        /// 从数据流中读取下一个int32
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        int Read32(Stream s)
        {
            var x = new byte[4];
            s.Read(x, 0, 4);
            return DataConverter.BigEndian.GetInt32(x, 0);
        }

        /// <summary>
        /// 处理图片数据
        /// </summary>
        /// <param name="input"></param>
        /// <param name="file"></param>
        /// <returns></returns>
        MnistImage[] ExtractImages(Stream input, string file)
        {
            //文件是gz格式的
            using (var gz = new GZipStream(input, CompressionMode.Decompress))
            {
                //不是2051说明下载的文件不对
                if (Read32(gz) != 2051)
                {
                    throw new Exception("不是2051说明下载的文件不对： " + file);
                }
                //图片数
                var count = Read32(gz);
                //行数
                var rows = Read32(gz);
                //列数
                var cols = Read32(gz);

                Console.WriteLine($"准备读取{count}张图片。");

                var result = new MnistImage[count];
                for (int i = 0; i < count; i++)
                {
                    //图片的大小（每个像素占一个bit)
                    var size = rows * cols;
                    var data = new byte[size];

                    //从数据流中读取这么大的一块内容
                    gz.Read(data, 0, size);

                    //将读取到的内容转换为MnistImage类型
                    result[i] = new MnistImage(cols, rows, data);
                }
                return result;
            }
        }

        /// <summary>
        /// 处理标签数据
        /// </summary>
        /// <param name="input"></param>
        /// <param name="file"></param>
        /// <returns></returns>
        byte[] ExtractLabels(Stream input, string file)
        {
            using (var gz = new GZipStream(input, CompressionMode.Decompress))
            {
                //不是2049说明下载的文件不对
                if (Read32(gz) != 2049)
                {
                    throw new Exception("不是2049说明下载的文件不对:" + file);
                }
                var count = Read32(gz);
                var labels = new byte[count];

                gz.Read(labels, 0, count);

                return labels;
            }
        }

        /// <summary>
        /// 获得source集合中的一部分，从first开始，到last结束
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <param name="first"></param>
        /// <param name="last"></param>
        /// <returns></returns>
        T[] Pick<T>(T[] source, int first, int last)
        {
            if (last == 0)
            {
                last = source.Length;
            }

            var count = last - first;
            var ret = source.Skip(first).Take(count).ToArray();
            return ret;
        }

        /// <summary>
        /// 将数字标签一维数组转为一个二维数组
        /// </summary>
        /// <param name="labels"></param>
        /// <param name="numClasses">多少个类别，这里是10（0到9）</param>
        /// <returns></returns>
        byte[,] OneHot(byte[] labels, int numClasses)
        {
            var oneHot = new byte[labels.Length, numClasses];
            for (int i = 0; i < labels.Length; i++)
            {
                oneHot[i, labels[i]] = 1;
            }
            return oneHot;
        }

        /// <summary>
        /// 处理数据集
        /// </summary>
        /// <param name="trainDir">数据集所在文件夹</param>
        /// <param name="numClasses"></param>
        /// <param name="validationSize">拿出多少做验证?</param>
        public void ReadDataSets(string trainDir, int numClasses = 10, int validationSize = 5000)
        {
            const string SourceUrl = "http://yann.lecun.com/exdb/mnist/";
            const string TrainImagesName = "train-images-idx3-ubyte.gz";
            const string TrainLabelsName = "train-labels-idx1-ubyte.gz";
            const string TestImagesName = "t10k-images-idx3-ubyte.gz";
            const string TestLabelsName = "t10k-labels-idx1-ubyte.gz";

            //获得训练数据，然后处理训练数据和测试数据
            TrainImages = ExtractImages(Helper.MaybeDownload(SourceUrl, trainDir, TrainImagesName), TrainImagesName);
            TestImages = ExtractImages(Helper.MaybeDownload(SourceUrl, trainDir, TestImagesName), TestImagesName);
            TrainLabels = ExtractLabels(Helper.MaybeDownload(SourceUrl, trainDir, TrainLabelsName), TrainLabelsName);
            TestLabels = ExtractLabels(Helper.MaybeDownload(SourceUrl, trainDir, TestLabelsName), TestLabelsName);

            //拿出前面的一部分做验证
            ValidationImages = Pick(TrainImages, 0, validationSize);
            ValidationLabels = Pick(TrainLabels, 0, validationSize);

            //拿出剩下的做训练（输入0意味着拿剩下所有的）
            TrainImages = Pick(TrainImages, validationSize, 0);
            TrainLabels = Pick(TrainLabels, validationSize, 0);

            //将数字标签转换为二维数组
            //例如，标签3 =》 [0,0,0,1,0,0,0,0,0,0]
            //标签0 =》 [1,0,0,0,0,0,0,0,0,0]
            if (numClasses != -1)
            {
                OneHotTrainLabels = OneHot(TrainLabels, numClasses);
                OneHotValidationLabels = OneHot(ValidationLabels, numClasses);
                OneHotTestLabels = OneHot(TestLabels, numClasses);
            }
        }

        public static Mnist Load()
        {
            var x = new Mnist();
            x.ReadDataSets(@"D:\人工智能\C#代码\MNISTTensorFlowSharp\MNISTTensorFlowSharp\data");
            return x;
        }
    }
}
