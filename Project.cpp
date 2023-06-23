#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>


using namespace std;
using namespace cv;
void Ransac(std::vector<cv::DMatch>& refined_matches,
            std::vector<cv::KeyPoint>& targetkeypoints,
            const cv::Mat& patch,
            cv::Mat& image,
            const std::vector<cv::KeyPoint>& keypoints);
void ManualRansac();

int main(int argc, char** argv)
{   // reading images and patches
    cv::Mat image = cv::imread("starwars/image_to_complete.jpg");

    cv::imshow("Corrupted Image", image);
    cv::waitKey(0);
    std::string path = "starwars/patches"; // Change the folder name patches_transform to try transformed patches
    cv::String file = path + "/*.jpg";  
    std::vector<cv::String> patches;
    cv::glob(file, patches);


    // Feature Extraction Part with SIFT 
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(); // SIFT function from Open CV
    std::vector<cv::KeyPoint> targetkeypoints;
    cv::Mat descriptors1;
    
    // detectandCompute and drawKeypoints for determine the features with sift
    sift->detectAndCompute(image, cv::noArray(), targetkeypoints, descriptors1); // from Open CV documentation

    cv::Mat image_keypoints;
    cv::Scalar keypointsColor(0, 255, 0);
    // draw the detected features
    cv::drawKeypoints(image, targetkeypoints, image_keypoints, keypointsColor, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);// from Open CV documentation
    
    
    // The original image without fills with feature extraction 
    cv::imshow("Corrupted Image with Keypoints", image_keypoints);
    cv::waitKey(0);


    // Brute-Force Matche from Open CV documentation 
    cv::BFMatcher matcher(cv::NORM_L2);// NORM_L2 is the best for SIFT, NORM_HAMMING best for ORB

    // Set a threshold ratio for matching, it determines threshold distance
    float threshold = 0.7f;

    // Overlay the patches over the image
    // for loop to match the all patches
    for (size_t i = 0; i < patches.size(); ++i)
    {
        cv::Mat patch = cv::imread(patches[i]);

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors2;
        sift->detectAndCompute(patch, cv::noArray(), keypoints, descriptors2);

        cv::Mat patch_keypoints;
        cv::Scalar keypointsColor(0, 255, 0);

        cv::drawKeypoints(patch, keypoints, patch_keypoints, keypointsColor, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        //cv::imshow("Corrupted Image with Keypoints", patch_keypoints);
        //cv::waitKey(0);

        // Match the features between the patch and image
        std::vector<std::vector<cv::DMatch>> matches;
        matcher.knnMatch(descriptors2, descriptors1, matches, 2); // k means neighour, 2 define the select 2 neighbor

        // Refine the matches based on the threshold ratio
        // select the important distance area features
        std::vector<cv::DMatch> refined_matches;
        for (size_t j = 0; j < matches.size(); ++j)
        {
            if (matches[j][0].distance < threshold * matches[j][1].distance)
            {
                refined_matches.push_back(matches[j][0]);
            }
        }
        // to show the matching lines between corrupted image and each patch 
        cv::Mat matched_image;
        cv::drawMatches(patch, keypoints, image, targetkeypoints, refined_matches, matched_image);

        cv::imshow("Matching Image", matched_image);
        cv::waitKey(0);

        Ransac(refined_matches, targetkeypoints, patch, image, keypoints);
        
    }
    ManualRansac();

    return 0;
}

void Ransac(std::vector<cv::DMatch>& refined_matches,
    std::vector<cv::KeyPoint>& targetkeypoints,
    const cv::Mat& patch,
    cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints) 
{
    // find the transformation between image features and patch features with RANSAC 
    std::vector<cv::Point2f> patch_matches;
    std::vector<cv::Point2f> image_matches;
    for (size_t j = 0; j < refined_matches.size(); ++j)
    {
        patch_matches.push_back(keypoints[refined_matches[j].queryIdx].pt);
        image_matches.push_back(targetkeypoints[refined_matches[j].trainIdx].pt);
    }

    cv::Mat homography = cv::findHomography(patch_matches, image_matches, cv::RANSAC); // Open CV documentation 

    // filled the corrupted image with proper patches
    cv::Mat filled_image = image.clone();
    cv::warpPerspective(patch, filled_image, homography, filled_image.size()); // Open CV Documentation 

    // set the patch with mask 
    cv::Mat mask;
    cv::cvtColor(filled_image, mask, cv::COLOR_BGR2GRAY);
    image.setTo(0, mask);

    // adding the pathes to the original image
    cv::add(image, filled_image, image);

    cv::imshow("Image with Overlayed Patch - RANSAC", image);
    cv::waitKey(0);

}


void ManualRansac()
{
    cv::Mat image = cv::imread("starwars/image_to_complete.jpg");

    cv::imshow("Corrupted Image", image);
    cv::waitKey(0);
    std::string path = "starwars/patches";
    cv::String file = path + "/*.jpg"; 
    std::vector<cv::String> patches;
    cv::glob(file, patches);


    // Extract SIFT features from the image
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> targetkeypoints;
    cv::Mat descriptors1;

    sift->detectAndCompute(image, cv::noArray(), targetkeypoints, descriptors1); 

    cv::Mat image_keypoints;
    cv::Scalar keypointsColor(0, 255, 0);

    cv::drawKeypoints(image, targetkeypoints, image_keypoints, keypointsColor, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
   
    cv::BFMatcher matcher(cv::NORM_L2);
    float threshold = 0.7;

    for (size_t i = 0; i < patches.size(); ++i)
    {
        cv::Mat patch = cv::imread(patches[i]);

        // SIFT PART
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors2;
        sift->detectAndCompute(patch, cv::noArray(), keypoints, descriptors2);

        cv::Mat patch_keypoints;
        cv::Scalar keypointsColor(0, 255, 0);

        cv::drawKeypoints(patch, keypoints, patch_keypoints, keypointsColor, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
       
        //cv::imshow("Corrupted Image with Keypoints", patch_keypoints);
        //cv::waitKey(0);

        std::vector<std::vector<cv::DMatch>> matches;
        matcher.knnMatch(descriptors2, descriptors1, matches, 2);

        std::vector<cv::DMatch> refined_matches;
        for (size_t j = 0; j < matches.size(); ++j)
        {
            if (matches[j][0].distance < threshold * matches[j][1].distance)
            {
                refined_matches.push_back(matches[j][0]);
            }
        }
 
        cv::Mat matched_image;
        cv::drawMatches(patch, keypoints, image, targetkeypoints, refined_matches, matched_image);

        
        cv::imshow("Matching Image- with Manual RANSAC", matched_image);
        cv::waitKey(0);
        // until here same with the above code
        // MANUAL RANSAC PART
        const int iterations = 1000;
        const double inlier_threshold = 3.0; // optimal value

        cv::RNG rng;
        cv::Mat bestAffineMatrix;
        int count = 0;

        // Select random 3 features to compare
        for (int i = 0; i < iterations; ++i) {
            
            std::set<int> randomIndices;
            while (randomIndices.size() < 3) {
                int index = rng.uniform(0, refined_matches.size());
                randomIndices.insert(index);
            }

            // finding the matches between patch and imaage based on selected random 3 feature
            std::vector<cv::Point2f> patch_matches;
            std::vector<cv::Point2f> image_matches;
            for (int index : randomIndices) {
                patch_matches.push_back(keypoints[refined_matches[index].queryIdx].pt);
                image_matches.push_back(targetkeypoints[refined_matches[index].trainIdx].pt);
            }

            // In the affine transform, matrix is calculated based on the selected matches
            cv::Mat affineMatrix = cv::getAffineTransform(patch_matches, image_matches);

            // the process continues until it reaches the threshold or iteration. 
            // between the affine matrixes based on selected random features, the best values affine matrix will selected as a transformation matrix
            int count_inlier = 0;
            for (const cv::DMatch& match : refined_matches) {
                cv::Point2f patchPoint = keypoints[match.queryIdx].pt;
                cv::Point2f imagePoint = targetkeypoints[match.trainIdx].pt;

                cv::Point2f transformedPoint = cv::Point2f(affineMatrix.at<double>(0, 0) * patchPoint.x + affineMatrix.at<double>(0, 1) * patchPoint.y + affineMatrix.at<double>(0, 2),
                    affineMatrix.at<double>(1, 0) * patchPoint.x + affineMatrix.at<double>(1, 1) * patchPoint.y + affineMatrix.at<double>(1, 2));

                double distance = cv::norm(transformedPoint - imagePoint);

                if (distance < inlier_threshold) {
                    ++count_inlier;
                }
            }

            // it updates the affine matrix, until it converges the best resukt or until the iteration ends 
            if (count_inlier > count) {
                count = count_inlier;
                bestAffineMatrix = affineMatrix;
            }
        }

        // Same as the first Part  
        cv::Mat filled_image = image.clone();
        cv::warpAffine(patch, filled_image, bestAffineMatrix, filled_image.size()); // rather than warpPerspective, we will use warpAffine. They basically do same thing
        cv::Mat mask;
        cv::cvtColor(filled_image, mask, cv::COLOR_BGR2GRAY);
        image.setTo(0, mask);
        cv::add(image, filled_image, image);
        cv::imshow("Image with Overlayed Patch - Manual RANSAC", image);
        cv::waitKey(0);

    }
    
}