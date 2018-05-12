using System.IO;
using System.Net;

namespace MNISTTensorFlowSharp
{
    public class Helper
    {
        /// <summary>
        /// 如果文件不存在就去下载
        /// </summary>
        /// <param name="urlBase">下载地址</param>
        /// <param name="trainDir">文件目录地址</param>
        /// <param name="file">文件名</param>
        /// <returns></returns>
        public static Stream MaybeDownload(string urlBase, string trainDir, string file)
        {
            if (!Directory.Exists(trainDir))
            {
                Directory.CreateDirectory(trainDir);
            }

            var target = Path.Combine(trainDir, file);
            if (!File.Exists(target))
            {
                var wc = new WebClient();
                wc.DownloadFile(urlBase + file, target);
            }
            return File.OpenRead(target);
        }
    }
}
