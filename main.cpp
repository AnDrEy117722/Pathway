#include <iostream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int, char**)
{
    Mat first, second, descr1, descr2, img_matches;

    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> extractor;
    BFMatcher matcher;

    vector<KeyPoint> keys1, keys2;
    vector<Point2d> VectorPoints; //Flight trajectory vector

    detector = BRISK::create();
    extractor = BRISK::create();

    // start point on the Graph
    double X = 500;
    double Y = 2000;
    VectorPoints.push_back({X,Y});


    VideoCapture cap("test.avi");
    if (!cap.isOpened())
        return -1;

    bool stop = false;
    // Определим частоту кадров на видео
    double rate = cap.get(CAP_PROP_FPS);
    // Рассчитаем задержку в миллисекундах
    int delay = 1000 / rate;

    cout << "Frame rate of video is " << rate << endl;

    double k = 0.6;

    //cap >> second;

    while (!stop)
    {
        // Read first frame
        bool result = cap.grab();
        if (result)
            cap >> first;
        else
            stop = true;

        // Read next frame
        result = cap.grab();
        if (result)
            cap >> second;
        else
            stop = true;

        detector->detect(first,keys1);
        detector->detect(second,keys2);
        extractor->compute(first, keys1, descr1);
        extractor->compute(second, keys2, descr2);

        //Knn match
        vector <vector<DMatch>> knn_matches;
        vector <DMatch> goodMatches;
        matcher.knnMatch(descr1, descr2, knn_matches, 2);
        for (auto item : knn_matches)
        {
            if (item[0].distance < k * item[1].distance) {
                goodMatches.push_back(item[0]);
            }
        }
        drawMatches(first, keys1, second, keys2, goodMatches, img_matches);
        imshow("KNN_matches", img_matches);

        vector<Point2f> FirstPoints, SecondPoints;
        for(size_t i = 0; i < goodMatches.size(); i++)
        {
            FirstPoints.push_back(keys1[goodMatches[i].queryIdx].pt);
            SecondPoints.push_back(keys2[goodMatches[i].trainIdx].pt);
        }

        Mat homography = estimateAffinePartial2D(FirstPoints, SecondPoints, noArray(), RANSAC, 3.0);
        invertAffineTransform(homography,homography);

        Point2d VectorPoint;
        X += homography.at<double>(0,2);
        Y += homography.at<double>(1,2);
        VectorPoint = {X, Y};
        VectorPoints.push_back(VectorPoint);

        // Ждём нажатия на кнопку
        int key = cv::waitKey(delay);
        if (key==27) {
            // Если это 0x27, т.е. ESC
            stop=true; // Выходим
        }

    }
    //see vector
    cout << VectorPoints.size() << endl;
    for (size_t i = 0; i < VectorPoints.size(); i++)
        cout << VectorPoints[i] << endl;

    //draw Graph
    Mat Graph = Mat::eye(Size(1000, 2000), CV_8UC3);
    for (size_t i = 0; i < VectorPoints.size() - 1; ++i)
        line(Graph, VectorPoints[i], VectorPoints[i+1], Scalar(0,255,0),3);
    imshow("Pathway", Graph);

    waitKey(0);
    return 0;
}
