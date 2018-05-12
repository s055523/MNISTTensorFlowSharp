using System;
using System.Collections.Generic;
using TensorFlow;

namespace MNISTTensorFlowSharp
{
    class Program
    {
        static void Main(string[] args)
        {
            //NearestNeighbor();
            //BasicOperation();
            //BasicPlaceholderOperation();
            //BasicMatrixOperation();
            KNN();

            Console.ReadKey();
        }

        //基础常量运算，演示了常量的使用
        static void BasicOperation()
        {
            using (var s = new TFSession())
            {
                var g = s.Graph;

                //建立两个TFOutput，都是常数
                var v1 = g.Const(1.5);
                var v2 = g.Const(0.5);

                //建立一个相加的运算
                var add = g.Add(v1, v2);

                //获得runner
                var runner = s.GetRunner();

                //相加
                var result = runner.Run(add);
                
                //获得result的值2
                Console.WriteLine($"相加的结果:{result.GetValue()}");
            }
        }

        //基础占位符运算
        static void BasicPlaceholderOperation()
        {
            using (var s = new TFSession())
            {
                var g = s.Graph;

                //占位符 - 一种不需要初始化，在运算时再提供值的对象
                //1*2的占位符
                var v1 = g.Placeholder(TFDataType.Double, new TFShape(2));
                var v2 = g.Placeholder(TFDataType.Double, new TFShape(2));

                //建立一个相乘的运算
                var add = g.Mul(v1, v2);

                //获得runner
                var runner = s.GetRunner();

                //相加
                //在这里给占位符提供值
                var data1 = new double[] { 0.3, 0.5 };
                var data2 = new double[] { 0.4, 0.8 };

                var result = runner
                    .Fetch(add)
                    .AddInput(v1, new TFTensor(data1))
                    .AddInput(v2, new TFTensor(data2))
                    .Run();

                var dataResult = (double[])result[0].GetValue();

                //获得result的值
                Console.WriteLine($"相乘的结果: [{dataResult[0]}, {dataResult[1]}]");
            }
        }

        //基础矩阵运算
        static void BasicMatrixOperation()
        {
            using (var s = new TFSession())
            {
                var g = s.Graph;

                //1x2矩阵
                var matrix1 = g.Const(new double[,] { { 1, 2 } });

                //2x1矩阵
                var matrix2 = g.Const(new double[,] { { 3 }, { 4 } });

                var product = g.MatMul(matrix1, matrix2);
                var result = s.GetRunner().Run(product);
                Console.WriteLine("矩阵相乘的值：" + ((double[,])result.GetValue())[0, 0]);
            };
        }

        //求两个点的L2距离
        static void DistanceL2(TFSession s, TFOutput v1, TFOutput v2)
        {
            var graph = s.Graph;

            //定义求距离的运算
            //这里要特别注意，如果第一个系数为double，第二个也需要是double，所以传入2d而不是2
            var pow = graph.Pow(graph.Sub(v1, v2), graph.Const(2d));

            //ReduceSum运算将输入的一串数字相加并得出一个值（而不是保留输入参数的size）
            var distance = graph.Sqrt(graph.ReduceSum(pow));

            //获得runner
            var runner = s.GetRunner();

            //求距离
            //在这里给占位符提供值
            var data1 = new double[] { 6, 4 };
            var data2 = new double[] { 9, 8 };

            var result = runner
                .Fetch(distance)
                .AddInput(v1, new TFTensor(data1))
                .AddInput(v2, new TFTensor(data2))
                .Run();

            Console.WriteLine($"点v1和v2的距离为{result[0].GetValue()}");
        }

        static void KNN()
        {
            //取得数据
            var mnist = Mnist.Load();

            //拿5000个训练数据，200个测试数据
            const int trainCount = 5000;
            const int testCount = 200;

            //获得的数据有两个
            //一个是图片，它们都是28*28的
            //一个是one-hot的标签，它们都是1*10的
            (var trainingImages, var trainingLabels) = mnist.GetTrainReader().NextBatch(trainCount);
            (var testImages, var testLabels) = mnist.GetTestReader().NextBatch(testCount);

            Console.WriteLine($"MNIST 1NN");

            //建立一个图表示计算任务
            using (var graph = new TFGraph())
            {
                var session = new TFSession(graph);

                //用来feed数据的占位符。trainingInput表示N张用来进行训练的图片,N是一个变量，所以这里使用-1
                TFOutput trainingInput = graph.Placeholder(TFDataType.Float, new TFShape(-1, 784));

                //xte表示一张用来测试的图片
                TFOutput xte = graph.Placeholder(TFDataType.Float, new TFShape(784));

                //计算这两张图片的L1距离。这很简单，实际上就是把784个数字逐对相减，然后取绝对值，最后加起来变成一个总和
                var distance = graph.ReduceSum(graph.Abs(graph.Sub(trainingInput, xte)), axis: graph.Const(1));

                //这里只是用了最近的那个数据
                //也就是说，最近的那个数据是什么，那pred（预测值）就是什么
                TFOutput pred = graph.ArgMin(distance, graph.Const(0));

                var accuracy = 0f;

                //开始循环进行计算，循环trainCount次
                for (int i = 0; i < testCount; i++)
                {
                    var runner = session.GetRunner();

                    //每次，对一张新的测试图，计算它和trainCount张训练图的距离，并获得最近的那张
                    var result = runner.Fetch(pred).Fetch(distance)
                        //trainCount张训练图（数据是trainingImages）
                        .AddInput(trainingInput, trainingImages)
                        //testCount张测试图（数据是从testImages中拿出来的）
                        .AddInput(xte, Extract(testImages, i))
                        .Run();
                    
                    //最近的点的序号
                    var nn_index = (int)(long)result[0].GetValue();

                    //从trainingLabels中找到答案（这是预测值）
                    var prediction = ArgMax(trainingLabels, nn_index);

                    //正确答案位于testLabels[i]中
                    var real = ArgMax(testLabels, i);

                    //PrintImage(testImages, i);

                    Console.WriteLine($"测试 {i}: " +
                        $"预测: {prediction} " +
                        $"正确答案: {real} (最近的点的序号={nn_index})");
                    //Console.WriteLine(testImages);

                    if (prediction == real)
                    {
                        accuracy += 1f / testCount;
                    }
                }
                Console.WriteLine("准确率: " + accuracy);

                session.CloseSession();
            }
        }

        /// <summary>
        /// 获取一个二位数组i[][]中，i[idx]的最大值（实际上就是把one-hot转回一个数字标签，以获得答案）
        /// </summary>
        /// <param name="array"></param>
        /// <param name="idx"></param>
        /// <returns></returns>
        static int ArgMax(float[,] array, int idx)
        {
            float max = -1;
            int maxIdx = -1;
            var l = array.GetLength(1);
            for (int i = 0; i < l; i++)
                if (array[idx, i] > max)
                {
                    maxIdx = i;
                    max = array[idx, i];
                }
            return maxIdx;
        }

        /// <summary>
        /// 从一个M*N的二维数组中获取N个值并输出为一个一维数组
        /// </summary>
        /// <param name="array"></param>
        /// <param name="index">从哪里开始</param>
        /// <returns></returns>
        public static float[] Extract(float[,] array, int index)
        {
            var n = array.GetLength(1);
            var ret = new float[n];

            for (int i = 0; i < n; i++)
                ret[i] = array[index, i];
            return ret;
        }

        public static void PrintImage(float[,] array, int i)
        {
            var l = array.GetLength(1);
            var currentLine = new List<string>();
            
            for (int j = 0; j < 784; j+=28)
            {
                currentLine.Clear();
                for (int k = j;k < j + 28; k++)
                {
                    var ret = array[i, k] * 255;
                    var str = ret.ToString().PadLeft(3, '0');
                    //if (str == "000") str = "   ";
                    currentLine.Add(str);
                }

                Console.WriteLine(string.Join(" ", currentLine));
            }
        }
    }
}
