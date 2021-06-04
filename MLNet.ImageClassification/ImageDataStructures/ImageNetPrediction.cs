using MLNet.ImageClassification.ModelScorer;
using Microsoft.ML.Data;

namespace MLNet.ImageClassification.ImageDataStructures
{
    public class ImageNetPrediction
    {
        [ColumnName(TFModelScorer.InceptionSettings.outputTensorName)]
        public float[] PredictedLabels;
    }
}
