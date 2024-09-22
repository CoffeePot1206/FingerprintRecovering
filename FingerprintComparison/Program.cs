using System;
using System.IO;
using System.Collections.Generic;
using SourceAFIS;
using SourceAFIS.Engine.Primitives;

class Program
{
    static void Main(string[] args)
    {
        string dataFolder = "./test_stain";
        string resultFile = "result.txt";

        // Lists to store scores
        List<double> oriScores = new List<double>();
        List<double> noisyScores = new List<double>();
        List<double> recoverScores = new List<double>();
        List<double> relativenoisyScores = new List<double>();
        List<double> relativerecoverScores = new List<double>();

        // Get all ori, noisy, and recover file pairs
        int index = 0;
        while (true)
        {
            string oriFile = Path.Combine(dataFolder, $"ori_{index}.png");
            string noisyFile = Path.Combine(dataFolder, $"noisy_{index}.png");
            string recoverFile = Path.Combine(dataFolder, $"recover_{index}.png");

            if (!File.Exists(oriFile) || !File.Exists(noisyFile) || !File.Exists(recoverFile))
            {
                break;
            }

            // Load fingerprint images
            FingerprintImage oriImage = new FingerprintImage(File.ReadAllBytes(oriFile));
            FingerprintImage noisyImage = new FingerprintImage(File.ReadAllBytes(noisyFile));
            FingerprintImage recoverImage = new FingerprintImage(File.ReadAllBytes(recoverFile));

            // Create templates
            FingerprintTemplate oriTemplate = new FingerprintTemplate(oriImage);
            FingerprintTemplate noisyTemplate = new FingerprintTemplate(noisyImage);
            FingerprintTemplate recoverTemplate = new FingerprintTemplate(recoverImage);

            //Compare ori with ori
            FingerprintMatcher matcher = new FingerprintMatcher(oriTemplate);
            double oriScore = matcher.Match(oriTemplate);
            oriScores.Add(oriScore);

            // Compare ori with noisy
            double noisyScore = matcher.Match(noisyTemplate);
            noisyScores.Add(noisyScore);

            // Compare ori with recover
            double recoverScore = matcher.Match(recoverTemplate);
            recoverScores.Add(recoverScore);

            // Relative score
            relativenoisyScores.Add(noisyScore / oriScore);
            relativerecoverScores.Add(recoverScore / oriScore);

            index++;
        }

        // Calculate average and variance
        double oriAverage = CalculateAverage(oriScores);
        double oriVariance = CalculateVariance(oriScores, oriAverage);

        double noisyAverage = CalculateAverage(noisyScores);
        double noisyVariance = CalculateVariance(noisyScores, noisyAverage);

        double recoverAverage = CalculateAverage(recoverScores);
        double recoverVariance = CalculateVariance(recoverScores, recoverAverage);

        double relativenoisyAverage = CalculateAverage(relativenoisyScores);
        double relativenoisyVariance = CalculateVariance(relativenoisyScores, relativenoisyAverage);

        double relativerecoverAverage = CalculateAverage(relativerecoverScores);
        double relativerecoverVariance = CalculateVariance(relativerecoverScores, relativerecoverAverage);

        // Write results to file
        using (StreamWriter writer = new StreamWriter(resultFile))
        {
            writer.WriteLine("Relative Noisy Scores:");
            writer.WriteLine($"Average: {relativenoisyAverage}");
            writer.WriteLine($"Variance: {relativenoisyVariance}");
            writer.WriteLine("Scores: " + string.Join(", ", relativenoisyScores));

            writer.WriteLine("\nRelative Recover Scores:");
            writer.WriteLine($"Average: {relativerecoverAverage}");
            writer.WriteLine($"Variance: {relativerecoverVariance}");
            writer.WriteLine("Scores: " + string.Join(", ", relativerecoverScores));

            writer.WriteLine("\nOriginal Scores:");
            writer.WriteLine($"Average: {oriAverage}");
            writer.WriteLine($"Variance: {oriVariance}");
            writer.WriteLine("Scores: " + string.Join(", ", oriScores));

            writer.WriteLine("\nNoisy Scores:");
            writer.WriteLine($"Average: {noisyAverage}");
            writer.WriteLine($"Variance: {noisyVariance}");
            writer.WriteLine("Scores: " + string.Join(", ", noisyScores));

            writer.WriteLine("\nRecover Scores:");
            writer.WriteLine($"Average: {recoverAverage}");
            writer.WriteLine($"Variance: {recoverVariance}");
            writer.WriteLine("Scores: " + string.Join(", ", recoverScores));
        }
    }

    static double CalculateAverage(List<double> scores)
    {
        double sum = 0;
        foreach (double score in scores)
        {
            sum += score;
        }
        return sum / scores.Count;
    }

    static double CalculateVariance(List<double> scores, double average)
    {
        double sum = 0;
        foreach (double score in scores)
        {
            sum += Math.Pow(score - average, 2);
        }
        return sum / scores.Count;
    }
}
