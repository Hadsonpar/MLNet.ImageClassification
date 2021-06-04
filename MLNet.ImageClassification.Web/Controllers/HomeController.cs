using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using MLNet.ImageClassification.Web.Models;
using System.IO;
using MLNet.ImageClassification.ModelScorer;
using Microsoft.AspNetCore.Http;
using MLNet.ImageClassification.ImageDataStructures;
using System.Drawing;

namespace MLNet.ImageClassification.Web.Controllers
{
    public class HomeController : Controller
    {
        public IActionResult Index()
        {
            return View();
        }
        [HttpPost]
        public async Task<IActionResult> Index(IFormFile imagec)
        {   
            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            var tagsTsv = Path.Combine(assetsPath, "inputs", "images", "tags.tsv");
            var imagesFolder = Path.Combine(assetsPath, "inputs", "images");
            var inceptionPb = Path.Combine(assetsPath, "inputs", "inception", "tensorflow_inception_graph.pb");
            var labelsTxt = Path.Combine(assetsPath, "inputs", "inception", "imagenet_comp_graph_label_strings.txt");

            ViewBag.PredictedLabel = null;
            ViewBag.Probability = null;
            ViewBag.PredictedMessage = null;

            try
            {
                var filePath = Path.GetTempFileName();
                //Save image
                using(var ms=new FileStream(filePath,FileMode.Create))
                {
                    await imagec.CopyToAsync(ms);
                }
                var modelScorer = new TFModelScorer(tagsTsv, imagesFolder, inceptionPb, labelsTxt);

                var prediction=(ImageNetDataProbability)modelScorer.Score(new ImageNetData() { ImagePath=filePath,Label=""});
                ViewBag.PredictedLabel = prediction.PredictedLabel;


                decimal valueProbability = (decimal)Math.Round(prediction.Probability, 2);
                ViewBag.Probability = valueProbability;

                if (valueProbability >= (decimal)Math.Round(0.80, 2) || valueProbability >= (decimal)Math.Round(1.00, 2))
                {
                    ViewBag.PredictedMessage = "S";
                }
                else if (valueProbability < (decimal)Math.Round(0.80, 2)) {
                    ViewBag.PredictedMessage = "N";
                }
            }
            catch
            {
                return View();
            }
            return View();
        }
        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;
            string fullPath = Path.Combine(assemblyFolderPath, relativePath);
            return fullPath;
        }

    }
}
