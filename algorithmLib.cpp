// This is the main DLL file.

#include "stdafx.h"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "algorithmLib.h"
#include <vector>
#include <time.h>
using namespace System;
using namespace cv;
using namespace std;
//2413,1028
Mat image00;
Mat imageGrey;
Mat imgROI;
Mat draw;
//Mat path_s;
float percentage;
float percentageRead;
Mat inImgTh;
Mat path;
float pc;
Mat src0;
Mat erosion_dst;
Mat  dilation_dst;
int erosion_elem = 0;
int erosion_size = 0;
int match_method;
Mat templ; Mat result;
char* image_window = "Source Image";
char* result_window = "Result window";

Mat z;
Mat defV;
Mat defH;

Point2f P1;
Point2f P2;



//------------------------
	vector< vector<Point> > contoursMask;
    vector<Vec4i> hierarchyMask;
//--------------------------

System::Drawing::Bitmap^ MatToBitmap(Mat srcImg){
	int stride = srcImg.size().width * srcImg.channels();//calc the srtide
	int hDataCount = srcImg.size().height;

	System::Drawing::Bitmap^ retImg;

	System::IntPtr ptr(srcImg.data);

	//create a pointer with Stride
	if (stride % 4 != 0){//is not stride a multiple of 4?
		//make it a multiple of 4 by fiiling an offset to the end of each row

		//to hold processed data
		uchar *dataPro = new uchar[((srcImg.size().width * srcImg.channels() + 3) & -4) * hDataCount];

		uchar *data = srcImg.ptr();

		//current position on the data array
		int curPosition = 0;
		//current offset
		int curOffset = 0;

		int offsetCounter = 0;

		//itterate through all the bytes on the structure
		for (int r = 0; r < hDataCount; r++){
			//fill the data
			for (int c = 0; c < stride; c++){
				curPosition = (r * stride) + c;

				dataPro[curPosition + curOffset] = data[curPosition];
			}

			//reset offset counter
			offsetCounter = stride;

			//fill the offset
			do{
				curOffset += 1;
				dataPro[curPosition + curOffset] = 0;

				offsetCounter += 1;
			} while (offsetCounter % 4 != 0);
		}

		ptr = (System::IntPtr)dataPro;//set the data pointer to new/modified data array

		//calc the stride to nearest number which is a multiply of 4
		stride = (srcImg.size().width * srcImg.channels() + 3) & -4;

		retImg = gcnew System::Drawing::Bitmap(srcImg.size().width, srcImg.size().height,
			stride,
			System::Drawing::Imaging::PixelFormat::Format24bppRgb,
			ptr);
	}
	else{

		//no need to add a padding or recalculate the stride
		retImg = gcnew System::Drawing::Bitmap(srcImg.size().width, srcImg.size().height,
			stride,
			System::Drawing::Imaging::PixelFormat::Format24bppRgb,
			ptr);
	}

	array<unsigned char, 1>^ imageData;
	System::Drawing::Bitmap^ output;

	// Create the byte array.
	{
		System::IO::MemoryStream^ ms = gcnew System::IO::MemoryStream();
		retImg->Save(ms, System::Drawing::Imaging::ImageFormat::Bmp);
		imageData = ms->ToArray();
		delete ms;
	}

	// Convert back to bitmap
	{
		System::IO::MemoryStream^ ms = gcnew System::IO::MemoryStream(imageData);
		output = (System::Drawing::Bitmap^)System::Drawing::Bitmap::FromStream(ms);
	}

	return output;
}
//bitmap to mat

Mat BitmapToMat(System::Drawing::Bitmap^ bitmap)
{

	System::Drawing::Rectangle blank = System::Drawing::Rectangle(0, 0, bitmap->Width, bitmap->Height);
	System::Drawing::Imaging::BitmapData^ bmpdata = bitmap->LockBits(blank, System::Drawing::Imaging::ImageLockMode::ReadWrite, bitmap->PixelFormat);
	if (bitmap->PixelFormat == System::Drawing::Imaging::PixelFormat::Format8bppIndexed)
	{
		//tmp = cvCreateImage(cvSize(bitmap->Width, bitmap->Height), IPL_DEPTH_8U, 1);
		//tmp->imageData = (char*)bmData->Scan0.ToPointer();
		cv::Mat thisimage(cv::Size(bitmap->Width, bitmap->Height), CV_8UC1, bmpdata->Scan0.ToPointer(), cv::Mat::AUTO_STEP);
		bitmap->UnlockBits(bmpdata);
		return thisimage;
	}

	else if (bitmap->PixelFormat == System::Drawing::Imaging::PixelFormat::Format24bppRgb)
	{

		cv::Mat thisimage(cv::Size(bitmap->Width, bitmap->Height), CV_8UC3, bmpdata->Scan0.ToPointer(), cv::Mat::AUTO_STEP);
		bitmap->UnlockBits(bmpdata);
		return thisimage;

	}

	Mat returnMat = (Mat::zeros(640, 480, CV_8UC1));
	//   bitmap->UnlockBits(bmData);
	return returnMat;
	//   return cvarrToMat(tmp);
}
//other funcs


int ptToVector(Point2i p, Point2i q,Point2f ( &returnVec)[2]) 
{
	//returnVec[2]----[0] element contains direction [1] contains point
    Point2f diff = p - q;
    float magV= sqrt(diff.x*diff.x + diff.y*diff.y);
	Mat img(640,480,CV_8UC3,Scalar(0,0,0));
	Point2f U=Point2f(diff.x/magV,diff.y/magV);
	//	char H[100];
	//sprintf_s(H,"diff.x=%3.2f  diff.y = %3.2f",diff.x,diff.y);
	////Point2f returnVec[2];
	//	char T[100];
	//sprintf_s(T,"U.x=%3.2f  U.y = %3.2f",U.x,U.y);
	//putText(img, H, Point(20, 40), CV_FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(155, 0, 0), 2, 4, false);
	//putText(img, T, Point(20, 120), CV_FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(155, 0, 0), 2, 4, false);
			//namedWindow( "circle_and_lines", WINDOW_AUTOSIZE );
			//  imshow("circle_and_lines",img);
			//  waitKey(0);
	returnVec[0]=Point2f(U.x,U.y);
	returnVec[1]=Point2f(p.x,p.y);
	return 1;
}

Point2f  getPointAtDist(Point2f returnVec[2], int distance,int position) {
	//returnVec[2]----[0] element contains direction [1] contains point
   Point2f returnPoint; 
   Point2f swappedVec;
   swappedVec.x=returnVec[0].y;
   swappedVec.y=returnVec[0].x;
	if (position ==1) //opposite direction
   {
	   returnPoint= returnVec[1] - distance* returnVec[0];
   }
   else if (position ==2) //90 CW invert Y
   {
      returnPoint= returnVec[1] + distance* Point2f(swappedVec.x,-1*swappedVec.y);
   }
   else if (position ==3) // 90  CCW invert x
   {
	  returnPoint= returnVec[1] + distance* Point2f(-1*swappedVec.x,swappedVec.y);
   }
   else //same direction
   {
   	   returnPoint= returnVec[1] + distance* returnVec[0];
   }
   return returnPoint;
}

float getAngle(Point l1s,Point l1e,Point l2s,Point l2e)
{
	float ang1=atan2(l1s.y-l1e.y,l1s.x-l1e.x);
	float ang2=atan2(l2s.y-l2e.y,l2s.x-l2e.x);
	float ang=(ang2-ang1)*180/(3.14);
	if (ang<0)
	{ang+=360;
	}
	return ang;
}

int algorithmLib::Class1::pathReader() {

	string str1 = "C:/Add Innovations/trainPath.txt", strOut;

	ifstream infileF(str1.c_str());

	if (infileF.is_open())
	{
		getline(infileF, strOut);
		replace(strOut.begin(), strOut.end(), '\\', '/');
		infileF.close();
	}

	//path = strOut;
	cout << "path::" << path << endl;
	return 1;
}

//img Processing

Mat src;
Mat src_gray;
Mat imgGrey;
int count = 0, j;
int counter = 0;
double largest_area = 0;
int largest_contour_index = 0;
int thresh;
int max_thresh;
RNG rng(12345);



 bool okNg=false;
	Point d1;
	Point d2;
	Point d3;



int out = 1;
int ringWidth=35;
Mat red;
Mat green;
Mat blue;


	
	Mat rz;
Mat Template_match_with_rotation_match0(Mat roi, Mat temp, Mat ImgToBeRotated, int rotaion_degree) //roi,temp ----image and template in any form i.e. edge,inrange etc, grayImagetoberotated--original image which will be rotated and returned
{
    Mat resz;
    resize(roi, resz, Size(), 0.5, 0.5);
    //		imshow("roi",resz);

    Mat roi2;
    int rotationAngles = rotaion_degree;
    resize(roi, roi2, cv::Size(roi.cols / 4, roi.rows / 4));
    Mat temp_roi2;
    resize(temp, temp_roi2, cv::Size(temp.cols / 4, temp.rows /4));
    // imshow("template",temp_roi2);
    //cv::Point2f center1(roi2.cols / 2.0, roi2.rows / 2.0);
    cv::Point2f center1(temp_roi2.cols / 2.0, temp_roi2.rows / 2.0);
    cv::Size a1 = cv::Size(temp_roi2.cols, temp_roi2.rows);
    vector<double> Minvalues(360);
    Mat roi_rot;
    int result_cols = roi2.cols - temp_roi2.cols + 1;
    int result_rows = roi2.rows - temp_roi2.rows + 1;
    Mat dstImage;
    dstImage.create(result_rows, result_cols, CV_32FC1);
    int match_method = 0; //0-4
    double matchTemp;
    double minVal = 1000000000000;

    double maxVal = 0;
    double maxValBest = 2;
    double angleBest = 0;
    double minValBest = 1000000000000;
    cv::Point minLoc;
    cv::Point maxLoc;
    double angleInc = 0;
    float indexer;
    for (int i = -1 * 2 * rotationAngles; i < 2 * rotationAngles; i++) {
        indexer = i;
        if (i == 0)
            angleInc = 0;
        else {
            angleInc = indexer;
        }

        //		 cout<<angleInc<<endl;
        Mat rot_mat = getRotationMatrix2D(center1, double(angleInc), 1.0);
        cv::Rect bbox = cv::RotatedRect(cv::Point2f(), temp_roi2.size(), double(angleInc)).boundingRect();
        // adjust transformation matrix
        rot_mat.at<double>(0, 2) += bbox.width / 2.0 - temp_roi2.cols / 2.0;
        rot_mat.at<double>(1, 2) += bbox.height / 2.0 - temp_roi2.rows / 2.0;
        // warpAffine(edgeImage2, edgeImage2_rot, rot_mat, a, 1);
        //warpAffine(roi2, roi_rot, rot_mat, a1, 1);
        Mat templRot;
        warpAffine(temp_roi2, templRot, rot_mat, bbox.size());

        //draw mask----------------
        Mat mask = Mat::zeros(templRot.size(), CV_8UC3);
        cv::RotatedRect rotatedRectangle(cv::Point2f(templRot.cols / 2, templRot.rows / 2), temp_roi2.size(), double(180 - angleInc));

        // We take the edges that OpenCV calculated for us
        cv::Point2f vertices2f[4];
        rotatedRectangle.points(vertices2f);

        // Convert them so we can use them in a fillConvexPoly
        cv::Point vertices[4];
        for (int i = 0; i < 4; ++i) {
            vertices[i] = vertices2f[i];
        }

        // Now we can fill the rotated rectangle with our specified color
        cv::fillConvexPoly(mask, vertices, 4, Scalar(255));
        //----------draw mask end

        cv::Point matchLoc;
      /*  cout<<"roi2vvvv---"<<roi2.size()<<endl;
        cout<<"templRotvvvv---"<<templRot.size()<<endl;

        cout << "roi2vvvv---" << roi2.channels() << endl;
        cout << "templRotvvvv---" << templRot.channels() << endl;
        cout << "mask---" << mask.channels() << endl;;*/

        /*   imshow("roi2wwe",roi2);
        imshow("templRotwwe",templRot);
        imshow("dstImage", dstImage);
        waitKey(0);*/
        matchTemplate(roi2, templRot, dstImage, match_method, mask);
        minMaxLoc(dstImage, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
        //Minvalues[i] = minVal;
        //rectangle(roi2, minLoc, cv::Point(minLoc.x + temp_roi2.cols, minLoc.y + temp_roi2.rows), Scalar(0, 0, 0), 2, 8, 0);
        //		imshow("rotImg",templRot);
        //		imshow("mask",mask);
        //	waitKey(0);

        //		cout<<"maxVal=========="<<maxVal<<endl;
        //		cout<<"angleInc=========="<<angleInc<<endl;

        if (minVal < minValBest) {
            minValBest = minVal;

            angleBest = angleInc;
        }

        // waitKey();
        // waitKey(0);
        //cout << "min_temp_value: "<< minVal2 << "\t" << i<<endl;
    }

    double min_temp_value = minValBest;
    double angle = -1 * angleBest;
    //*angleRet = angleBest;

    //	cout<<"min_temp_value=========="<<min_temp_value<<endl;
    //	cout<<"angle=========="<<angle<<endl;

    //	 cout<<"Angle"<<angleBest<<endl;
    //	 cout<<"Value"<<minValBest<<endl;
    //for (int i = 0; i < rotationAngles ; i++ )
    //{
    // if(Minvalues[i]  < min_temp_value)
    // {
    //	 min_temp_value = Minvalues[i] ;
    //	 angle = i;
    // }
    //}
    //  cout << "angle: " << angle <<endl;
    // cout << "min_temp_value: "<< min_temp_value <<endl;
    cv::Point2f center2(roi.cols / 2.0, roi.rows / 2.0);
    cv::Size a2 = cv::Size(roi.cols, roi.rows);
    cv::Size a3 = cv::Size(ImgToBeRotated.cols, ImgToBeRotated.rows);
    Mat rot_mat = getRotationMatrix2D(center2, angle, 1.0);

    // get rotation matrix for rotating the image around its center in pixel coordinates
    // determine bounding rectangle, center not relevant
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), roi.size(), angle).boundingRect();
    // adjust transformation matrix
    rot_mat.at<double>(0, 2) += bbox.width / 2.0 - roi.cols / 2.0;
    rot_mat.at<double>(1, 2) += bbox.height / 2.0 - roi.rows / 2.0;

    warpAffine(roi, roi_rot, rot_mat, a2, 1);
    Mat returnImage;
    warpAffine(ImgToBeRotated, returnImage, rot_mat, bbox.size());
    matchTemplate(roi_rot, temp, dstImage, match_method);
    minMaxLoc(dstImage, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    //	*matchCoord = maxLoc;
    //	*matchValue = maxVal;
    //	cout<<"match value------------"<<maxVal<<endl;
    //	imwrite("C:/Users/VisionSystem/Pictures/Saved Pictures/new rot.bmp",returnImage);

    //	resize(returnImage,resz,Size(),0.5,0.5);
  /*  Rect r1 = Rect(Point(minLoc.x, minLoc.y),Point(minLoc.x + temp.cols, minLoc.y + (temp.rows+900)));
    Mat prnt_crp = ImgToBeRotated(r1);*/


    Rect r2 = Rect(minLoc, Point(minLoc.x + temp.cols, minLoc.y + temp.rows));
    Mat prnt_crpxx = returnImage(r2);


    resize(prnt_crpxx, rz, Size(), 0.3, 0.3);
  //  imshow("prnt_crpxx", rz);


   // waitKey(0);
    return prnt_crpxx;
}

Mat key_crop;

int pix_val = 0;
vector<Point>blck_point;
vector<Point>blck_point_t2b;
vector<Point>blck_strt_point;
vector<Point> bigcont;
Point min_Y;
Point max_Y;
int mov_along_cols_top2bottom(Point a, Point b, int width, Mat inimg4, Mat thrs_img/*, Point* min, Point* max*/)
{
    for (int i = a.x; i < (b.x) - 1; i++)
    {
        for (int j = (a.y); j < (b.y) ; j++)
        {
            //cout << "11111====" << endl;

            //	imshow("thrs_img", thrs_img);
            //
            pix_val = thrs_img.at<uchar>(j, i);
            //	cout << " (a.x)-width:::::::::::" << (a.x) - width << endl;
            //cout << "  a.y:::::::::::" << a.y << endl;
            /*cout << "  pix_val------------ " << pix_val << endl;

            imshow("inimg4", inimg4);
            waitKey(0);*/
            //	cout << "  pix_val------------ " << pix_val << endl;
            if (pix_val == 0)
            {
              //  circle(inimg4, Point(i,j),2, Scalar(255, 0, 0), -1);
                blck_point_t2b.push_back(Point(i, j));

              
               // waitKey(1);
                //break;
            }
            else
            {
             //   circle(inimg4, Point(i, j), 2, Scalar(0, 0, 255), -1);

            }
            
        }
       
    }

    int n = blck_point_t2b.size();
    cout << blck_point_t2b[n-1] << endl;


    //*strt_point = blck_point_t2b[0];
    //*end_point = blck_point_t2b[n - 1];


    bigcont = blck_point_t2b;
    min_Y = Point(0, 1000);
    max_Y = Point(0, 0);
    //	cout << "blck_point_t2b::::::::" << blck_point_t2b << endl;
    for (int j = 0; j < bigcont.size(); j++)
    {


        if (min_Y.y > bigcont[j].y)
        {
            min_Y = bigcont[j];
        }

        if (max_Y.y < bigcont[j].y)
        {
            max_Y = bigcont[j];
        }
    }
    cout << "max.x:::::::::" << max_Y << endl;
    cout << "min.x::::::::" << min_Y << endl;
   // *min = min_Y;
   // *max = max_Y;
    circle(inimg4, max_Y, 5, Scalar(255, 0, 0), -1);
    circle(inimg4, min_Y, 5, Scalar(255, 0, 0), -1);







  //  cout << "blck_point_t2b===" << blck_point_t2b << endl;
    int countr=0;
    for (int k = 0; k < blck_point_t2b.size() - 1; k = k + 1)
    {
        //if(blck_point[k].x>blck_point[k+1].x)

        //	cout<<"diffff----"<<abs((blck_point[k+1].x-blck_point[k].x) )<<endl;

        int pix_diff = abs((blck_point_t2b[k+1].y - blck_point_t2b[k].y));

        if (abs(blck_point_t2b[k+1].y - blck_point_t2b[k].y)>50 /*|| pix_diff > 2*/)
        {
            //	line(inimg4,blck_point[k],blck_point[k+1],Scalar(0,250,0),6);
            circle(inimg4, Point(blck_point_t2b[k].x, blck_point_t2b[k].y), 2, Scalar(0, 250, 0), -1);

          //  cout << "sssssssss" << endl;
            countr++;
        }
    }

    cout << "countr----------" << countr << endl;




   // resize(inimg4, rz, Size(), 0.3, 0.3);
   // imshow("inimg4", inimg4);
   // waitKey(0);
    blck_point_t2b.clear();
    return 1;
}

Mat Template_match_with_rotation_match1(Mat roi, Mat temp, Mat ImgToBeRotated, int rotaion_degree) //roi,temp ----image and template in any form i.e. edge,inrange etc, grayImagetoberotated--original image which will be rotated and returned
{
    Mat resz;
    resize(roi, resz, Size(), 0.5, 0.5);
    //		imshow("roi",resz);

    Mat roi2_B;
    Mat temp_B;

    cvtColor(roi, roi2_B, COLOR_BGR2GRAY);
    // cvtColor(mask, mask, COLOR_BGR2GRAY);
    cvtColor(temp, temp_B, COLOR_BGR2GRAY);

    threshold(roi2_B, roi2_B, 150, 255, THRESH_BINARY);
    threshold(temp_B, temp_B, 150, 255, THRESH_BINARY);





    Mat roi2;
    int rotationAngles = rotaion_degree;
    resize(roi2_B, roi2, cv::Size(roi.cols , roi.rows ));
    Mat temp_roi2;
    resize(temp_B, temp_roi2, cv::Size(temp.cols , temp.rows ));

    // imshow("template",temp_roi2);
    //cv::Point2f center1(roi2.cols / 2.0, roi2.rows / 2.0);
    cv::Point2f center1(temp_roi2.cols / 2.0, temp_roi2.rows / 2.0);
    cv::Size a1 = cv::Size(temp_roi2.cols, temp_roi2.rows);
    vector<double> Minvalues(360);
    Mat roi_rot;
    int result_cols = roi2.cols - temp_roi2.cols + 1;
    int result_rows = roi2.rows - temp_roi2.rows + 1;
    Mat dstImage;
    dstImage.create(result_rows, result_cols, CV_32FC1);
    int match_method = 0; //0-4
    double matchTemp;
    double minVal = 1000000000000;

    double maxVal = 0;
    double maxValBest = 2;
    double angleBest = 0;
    double minValBest = 1000000000000;
    cv::Point minLoc;
    cv::Point maxLoc;
    double angleInc = 0;
    float indexer;
    for (int i = -1 * 2 * rotationAngles; i < 2 * rotationAngles; i++) {
        indexer = i;
        if (i == 0)
            angleInc = 0;
        else {
            angleInc = indexer;
        }

        //		 cout<<angleInc<<endl;
        Mat rot_mat = getRotationMatrix2D(center1, double(angleInc), 1.0);
        cv::Rect bbox = cv::RotatedRect(cv::Point2f(), temp_roi2.size(), double(angleInc)).boundingRect();
        // adjust transformation matrix
        rot_mat.at<double>(0, 2) += bbox.width / 2.0 - temp_roi2.cols / 2.0;
        rot_mat.at<double>(1, 2) += bbox.height / 2.0 - temp_roi2.rows / 2.0;
        // warpAffine(edgeImage2, edgeImage2_rot, rot_mat, a, 1);
        //warpAffine(roi2, roi_rot, rot_mat, a1, 1);
        Mat templRot;
        warpAffine(temp_roi2, templRot, rot_mat, bbox.size());

        //draw mask----------------
        Mat mask = Mat::zeros(templRot.size(), CV_8UC1);
        cv::RotatedRect rotatedRectangle(cv::Point2f(templRot.cols / 2, templRot.rows / 2), temp_roi2.size(), double(180 - angleInc));

        // We take the edges that OpenCV calculated for us
        cv::Point2f vertices2f[4];
        rotatedRectangle.points(vertices2f);

        // Convert them so we can use them in a fillConvexPoly
        cv::Point vertices[4];
        for (int i = 0; i < 4; ++i) {
            vertices[i] = vertices2f[i];
        }

        // Now we can fill the rotated rectangle with our specified color
        cv::fillConvexPoly(mask, vertices, 4, Scalar(255,255,255));
        //----------draw mask end

        cv::Point matchLoc;
             //cout<<"roi2vvvv---"<<roi2.size()<<endl;
             //cout<<"templRotvvvv---"<<templRot.size()<<endl;

             //cout << "roi2vvvv---" << roi2.channels() << endl;
             //cout << "templRotvvvv---" << templRot.channels() << endl;
             //cout << "mask---" << mask.channels() << endl;;

       

             //imshow("roi2wwe",roi2);
             //imshow("templRotwwe",templRot);
             ////imshow("dstImage", dstImage);
             //imshow("mask", mask);
             //waitKey(5);
        matchTemplate(roi2, templRot, dstImage, match_method, mask);

        minMaxLoc(dstImage, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
        //Minvalues[i] = minVal;
        //rectangle(roi2, minLoc, cv::Point(minLoc.x + temp_roi2.cols, minLoc.y + temp_roi2.rows), Scalar(0, 0, 0), 2, 8, 0);
        //		imshow("rotImg",templRot);
        //		imshow("mask",mask);
        //	waitKey(0);

        //		cout<<"maxVal=========="<<maxVal<<endl;
        //		cout<<"angleInc=========="<<angleInc<<endl;

        if (minVal < minValBest) {
            minValBest = minVal;

            angleBest = angleInc;
        }

        // waitKey();
        // waitKey(0);
        //cout << "min_temp_value: "<< minVal2 << "\t" << i<<endl;
    }

    double min_temp_value = minValBest;
    double angle = -1 * angleBest;
    //*angleRet = angleBest;

    //	cout<<"min_temp_value=========="<<min_temp_value<<endl;
   	cout<<"angle=========="<<angle<<endl;

    //	 cout<<"Angle"<<angleBest<<endl;
    //	 cout<<"Value"<<minValBest<<endl;
    //for (int i = 0; i < rotationAngles ; i++ )
    //{
    // if(Minvalues[i]  < min_temp_value)
    // {
    //	 min_temp_value = Minvalues[i] ;
    //	 angle = i;
    // }
    //}
    //  cout << "angle: " << angle <<endl;
    // cout << "min_temp_value: "<< min_temp_value <<endl;
    cv::Point2f center2(roi.cols / 2.0, roi.rows / 2.0);
    cv::Size a2 = cv::Size(roi.cols, roi.rows);
    cv::Size a3 = cv::Size(ImgToBeRotated.cols, ImgToBeRotated.rows);
    Mat rot_mat = getRotationMatrix2D(center2, angle, 1.0);

    // get rotation matrix for rotating the image around its center in pixel coordinates
    // determine bounding rectangle, center not relevant
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), roi.size(), angle).boundingRect();
    // adjust transformation matrix
    rot_mat.at<double>(0, 2) += bbox.width / 2.0 - roi.cols / 2.0;
    rot_mat.at<double>(1, 2) += bbox.height / 2.0 - roi.rows / 2.0;

    warpAffine(roi, roi_rot, rot_mat, a2, 1);
    Mat returnImage;
    warpAffine(ImgToBeRotated, returnImage, rot_mat, bbox.size());
    matchTemplate(roi_rot, temp, dstImage, match_method);
    minMaxLoc(dstImage, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    
    //	*matchCoord = maxLoc;
    //	*matchValue = maxVal;
    //	cout<<"match value------------"<<maxVal<<endl;
    //	imwrite("C:/Users/VisionSystem/Pictures/Saved Pictures/new rot.bmp",returnImage);

    //	resize(returnImage,resz,Size(),0.5,0.5);
    

   // rectangle(returnImage, minLoc, Point(minLoc.x + temp.cols, minLoc.y + temp.rows), Scalar::all(0), 2, 8, 0);
    
    Rect r2 = Rect(minLoc, Point(minLoc.x + temp.cols, minLoc.y + temp.rows));
    Mat prnt_crpxx = returnImage(r2);


    resize(prnt_crpxx, rz, Size(), 0.3, 0.3);
  //  imshow("prnt_crpxx", rz);

    
    return prnt_crpxx;
}


Mat similarity_chck(Mat rotated1,int *cn)
{
	Rect x1=Rect(161,129,329,137);
    Mat img1 = rotated1(x1);
  //  Mat img1 = rotated1(Rect(1570, 655, 390, 145));
    Mat img1_gray;
    cvtColor(img1,img1_gray,COLOR_BGR2GRAY);

    Mat img1_th;

    threshold(img1_gray,img1_th,150,255,CV_THRESH_BINARY_INV);
    resize(img1_th, rz, Size(), 0.3, 0.3);
  
    vector<vector<Point> > contour;

    vector<Vec4i> hierarcy;

    findContours(img1_th, contour, hierarcy,RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contour.size(); i++)
    {
        RotatedRect a1 = minAreaRect(contour[i]);
       int area = contourArea(contour[i], false);

     

       if (area>100)
       {
           Rect boundrect = boundingRect(contour[i]);
           int width1 = boundrect.width;
           int height1 = boundrect.height;

         //  cout << "area:::::::::::" << area << endl;
         //  drawContours(img1, contour, i, Scalar(0, 0, 250), 3, 8);
         //  rectangle(img1, boundrect.tl(), boundrect.br(), Scalar(255,0,0), 2);
           Rect x1 = Rect(boundrect.tl(), boundrect.br());
            key_crop = img1(x1);
          // resize(key_crop, rz, Size(), 0.3, 0.3);
         //  imshow("key_crop", key_crop);
       }
      
    }
   // imshow("imgg1111", img1);
    

    Rect x2 = Rect(77, 429, 429, 273);
  //  Rect x2 = Rect(1470, 925, 563, 409);
    Mat testing_img = rotated1(x2);
 //   imshow("testing_img", testing_img);
  //  waitKey(0);
    Mat img_rot = testing_img.clone();
    Mat testing_img_T;
   

  Mat final_img=  Template_match_with_rotation_match1(testing_img, key_crop, img_rot, 10);


 // imshow("final_img", final_img);


  Mat diff_img;
  absdiff(final_img, key_crop, diff_img);
  cvtColor(diff_img, diff_img,COLOR_BGR2GRAY);
  threshold(diff_img, diff_img,150,255,THRESH_BINARY);

 //  imshow("diff_img", diff_img);

  
   rectangle(rotated1,x1,Scalar(255,0,0),2,2);
     rectangle(rotated1,x2,Scalar(0,0,250),2,2);

   float cnz=countNonZero(diff_img);
   cout << "cnz:::::::::" << cnz << endl;
   *cn=cnz;
  //  resize(rotated1, rz, Size(), 0.3, 0.3);
 //   imshow("rotated1rotated1", rz);
 // waitKey(0);

  

   // resize(img1, rz, Size(), 0.3, 0.3);
  //  imshow("imgg1111", rz);
    return(rotated1);
}



Mat similarity_chck2(Mat rotated1,int *cn)
{
	Rect x1=Rect(1225, 547, 267, 103);
    Mat img1 = rotated1(x1);
    //  Mat img1 = rotated1(Rect(1570, 655, 390, 145));
    Mat img1_gray;
    cvtColor(img1, img1_gray, COLOR_BGR2GRAY);

    Mat img1_th;

    threshold(img1_gray, img1_th, 150, 255, CV_THRESH_BINARY_INV);
    resize(img1_th, rz, Size(), 0.3, 0.3);

    vector<vector<Point> > contour;

    vector<Vec4i> hierarcy;

    findContours(img1_th, contour, hierarcy, RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contour.size(); i++)
    {
        RotatedRect a1 = minAreaRect(contour[i]);
        int area = contourArea(contour[i], false);



        if (area > 100)
        {
            Rect boundrect = boundingRect(contour[i]);
            int width1 = boundrect.width;
            int height1 = boundrect.height;

            //  cout << "area:::::::::::" << area << endl;
            //  drawContours(img1, contour, i, Scalar(0, 0, 250), 3, 8);
            //  rectangle(img1, boundrect.tl(), boundrect.br(), Scalar(255,0,0), 2);
            Rect x1 = Rect(boundrect.tl(), boundrect.br());
            key_crop = img1(x1);
            // resize(key_crop, rz, Size(), 0.3, 0.3);
         //   imshow("key_crop", key_crop);
        }

    }
    // imshow("imgg1111", img1);


    Rect x2 = Rect(1161, 811, 371, 243);
    //  Rect x2 = Rect(1470, 925, 563, 409);
    Mat testing_img = rotated1(x2);
  //  imshow("testing_img", testing_img);
  //  waitKey(0);
    Mat img_rot = testing_img.clone();
    Mat testing_img_T;


    Mat final_img = Template_match_with_rotation_match1(testing_img, key_crop, img_rot, 10);


  //  imshow("final_img", final_img);


    Mat diff_img;
    absdiff(final_img, key_crop, diff_img);
    cvtColor(diff_img, diff_img, COLOR_BGR2GRAY);
    threshold(diff_img, diff_img, 150, 255, THRESH_BINARY);

 //   imshow("diff_img", diff_img);


    float cnz = countNonZero(diff_img);
    cout << "cnz:::::::::" << cnz << endl;
	*cn =cnz;
	

	 rectangle(rotated1,x1,Scalar(255,0,0),2,2);
     rectangle(rotated1,x2,Scalar(0,0,250),2,2);

    return(rotated1);
}



Mat similarity_chck3(Mat rotated1,int *cn)
{
	Rect x1=Rect(95, 863, 319, 107);
    Mat img1 = rotated1(x1);
    //  Mat img1 = rotated1(Rect(1570, 655, 390, 145));
    Mat img1_gray;
    cvtColor(img1, img1_gray, COLOR_BGR2GRAY);

    Mat img1_th;

    threshold(img1_gray, img1_th, 150, 255, CV_THRESH_BINARY_INV);
    resize(img1_th, rz, Size(), 0.3, 0.3);

    vector<vector<Point> > contour;

    vector<Vec4i> hierarcy;

    findContours(img1_th, contour, hierarcy, RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contour.size(); i++)
    {
        RotatedRect a1 = minAreaRect(contour[i]);
        int area = contourArea(contour[i], false);



        if (area > 100)
        {
            Rect boundrect = boundingRect(contour[i]);
            int width1 = boundrect.width;
            int height1 = boundrect.height;

            //  cout << "area:::::::::::" << area << endl;
            //  drawContours(img1, contour, i, Scalar(0, 0, 250), 3, 8);
            //  rectangle(img1, boundrect.tl(), boundrect.br(), Scalar(255,0,0), 2);
            Rect x1 = Rect(boundrect.tl(), boundrect.br());
            key_crop = img1(x1);
            // resize(key_crop, rz, Size(), 0.3, 0.3);
         //   imshow("key_crop", key_crop);
        }

    }
    // imshow("imgg1111", img1);


    Rect x2 = Rect(99, 1169, 331, 259);
    //  Rect x2 = Rect(1470, 925, 563, 409);
    Mat testing_img = rotated1(x2);
  //  imshow("testing_img", testing_img);
  //  waitKey(0);
    Mat img_rot = testing_img.clone();
    Mat testing_img_T;


    Mat final_img = Template_match_with_rotation_match1(testing_img, key_crop, img_rot, 10);


  //  imshow("final_img", final_img);


    Mat diff_img;
    absdiff(final_img, key_crop, diff_img);
    cvtColor(diff_img, diff_img, COLOR_BGR2GRAY);
    threshold(diff_img, diff_img, 150, 255, THRESH_BINARY);

 //   imshow("diff_img", diff_img);


    float cnz = countNonZero(diff_img);
    cout << "cnz:::::::::" << cnz << endl;
	*cn=cnz;
	

	 rectangle(rotated1,x1,Scalar(255,0,0),2,2);
     rectangle(rotated1,x2,Scalar(0,0,250),2,2);

    return(rotated1);
}



Mat similarity_chck4(Mat rotated1,int *cn)
{
	Rect x1=Rect(1219, 197, 305, 121);
    Mat img1 = rotated1(x1);
    //  Mat img1 = rotated1(Rect(1570, 655, 390, 145));
    Mat img1_gray;
    cvtColor(img1, img1_gray, COLOR_BGR2GRAY);

    Mat img1_th;

    threshold(img1_gray, img1_th, 150, 255, CV_THRESH_BINARY_INV);
    resize(img1_th, rz, Size(), 0.3, 0.3);

    vector<vector<Point> > contour;

    vector<Vec4i> hierarcy;

    findContours(img1_th, contour, hierarcy, RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contour.size(); i++)
    {
        RotatedRect a1 = minAreaRect(contour[i]);
        int area = contourArea(contour[i], false);



        if (area > 100)
        {
            Rect boundrect = boundingRect(contour[i]);
            int width1 = boundrect.width;
            int height1 = boundrect.height;

            //  cout << "area:::::::::::" << area << endl;
            //  drawContours(img1, contour, i, Scalar(0, 0, 250), 3, 8);
            //  rectangle(img1, boundrect.tl(), boundrect.br(), Scalar(255,0,0), 2);
            Rect x1 = Rect(boundrect.tl(), boundrect.br());
            key_crop = img1(x1);
            // resize(key_crop, rz, Size(), 0.3, 0.3);
         //   imshow("key_crop", key_crop);
        }

    }
    // imshow("imgg1111", img1);


    Rect x2 = Rect(1171, 501, 387, 255);
    //  Rect x2 = Rect(1470, 925, 563, 409);
    Mat testing_img = rotated1(x2);
  //  imshow("testing_img", testing_img);
  //  waitKey(0);
    Mat img_rot = testing_img.clone();
    Mat testing_img_T;


    Mat final_img = Template_match_with_rotation_match1(testing_img, key_crop, img_rot, 10);


  //  imshow("final_img", final_img);


    Mat diff_img;
    absdiff(final_img, key_crop, diff_img);
    cvtColor(diff_img, diff_img, COLOR_BGR2GRAY);
    threshold(diff_img, diff_img, 150, 255, THRESH_BINARY);

 //   imshow("diff_img", diff_img);


    float cnz = countNonZero(diff_img);
    cout << "cnz:::::::::" << cnz << endl;
	*cn=cnz;
	

	 rectangle(rotated1,x1,Scalar(255,0,0),2,2);
     rectangle(rotated1,x2,Scalar(0,0,250),2,2);

    return(rotated1);
}



	



	bool algorithmLib::Class1:: jaquarM2(int maskEn,System::Drawing::Bitmap^ bitmap0)
	{
		//string Path="D:/CV/08 august/New folder (2)/circle_ng/";
    //string Path="D:/CV/08 august/New folder (2)/circle_ng/";
    out = 0;
    // try {
    clock_t tStart = clock();
    /*	imwrite("1.bmp",BitmapToMat (bitmap0));
Mat img=imread("1.bmp");*/

    Mat img = BitmapToMat(bitmap0);
    //Mat bgr[3];
    //split(img,bgr);
    //if (maskEn==0)
    // red=bgr[0].clone();
    //else if (maskEn==1)
    //red=bgr[1].clone();
    //else
    //red=bgr[2].clone();
    ////imshow("red",bgr[2]);
    ////waitKey(0);
    //cvtColor(red,red,CV_GRAY2BGR);

    Mat temp = imread("F:/baumer img/key/temp.jpg");
    resize(temp, rz, Size(), 0.3, 0.3);
  //  imshow("temp", rz);
    //  waitKey(0);
    int rot = 1;
    Mat img_rot = img.clone();
    Mat rotated = Template_match_with_rotation_match0(img, temp, img_rot, rot);

	int cn;
    Mat outimg = similarity_chck4(rotated,&cn);
	Mat rz;
	resize(outimg,rz,img.size());
	rz.copyTo(img);

	bool rslt;
	if(cn>700)
	{
		putText(img, "Key Missmatch",Point(300,100),FONT_HERSHEY_DUPLEX,2,Scalar(0,0,255),2);
	rslt=0;
	}
	else
	{
	putText(img, "Key match",Point(300,100),FONT_HERSHEY_DUPLEX,2,Scalar(0,250,0),2);
	rslt=1;
	}

//	imshow("img",img);
//	waitKey(0);
    cout << "time taken::::::::" << double(clock() - tStart) / CLOCKS_PER_SEC;
    //------------------------------------------------------------------------------

    //-------------------------------------------------------------------------
    //cvtColor(imgRet,imgRet,CV_GRAY2BGR);
   // System::Drawing::Bitmap ^ dst = MatToBitmap(img);
    //return dst;
    //-----------------------------------------------------------------------

    //return dst;
    ////}
    ////catch( exception ex)
    ////{
    //return bitmap0;
    ////}
	return rslt;
	}





	bool algorithmLib::Class1:: jaquarM21(int maskEn,System::Drawing::Bitmap^ bitmap0)
	{
		//string Path="D:/CV/08 august/New folder (2)/circle_ng/";
  	//string Path="D:/CV/08 august/New folder (2)/circle_ng/";
    out = 0;
   // try {
          clock_t tStart=clock();
	
		   Mat img = BitmapToMat(bitmap0);

		   Mat temp = imread("F:/baumer img/key/temp.jpg");
    resize(temp, rz, Size(), 0.3, 0.3);
 //   imshow("temp", rz);
    //  waitKey(0);
    int rot = 1;
    Mat img_rot = img.clone();
    Mat rotated = Template_match_with_rotation_match0(img, temp, img_rot, rot);
	int cn;
    Mat outimg = similarity_chck2(rotated,&cn);
	Mat rz;
	resize(outimg,rz,img.size());
	rz.copyTo(img);
	bool rslt;

	//imshow("img",img);
	//waitKey(0);
    cout << "time taken::::::::" << double(clock() - tStart) / CLOCKS_PER_SEC;


	
	if(cn>700)
	{
		putText(img, "Key Missmatch",Point(300,100),FONT_HERSHEY_DUPLEX,2,Scalar(0,0,255),2);
	rslt=0;
	}
	else
	{
	putText(img, "Key match",Point(300,100),FONT_HERSHEY_DUPLEX,2,Scalar(0,250,0),2);
	rslt=1;
	}


	return rslt;
//}
	}



	Mat temp_match_with_rotation1(Mat inimg, Mat templ1)
{

    Mat grayImgToBeRotated = inimg.clone();
    Mat img1 = inimg.clone();
    //		imshow("thimg11111",img);

    Mat img;

    if (img1.channels() > 2) {
        cvtColor(img1, img, CV_BGR2GRAY);
    }
    else {
        img = img1.clone();
    }
    Mat imgth1;

    threshold(img, imgth1, 30, 255, THRESH_BINARY);

    Mat roi = imgth1.clone();

    //cout<<"chnl+++++++++"<<roi.channels()<<endl;
    //cout<<"chnl:::::::"<<temp.channels()<<endl;
    Mat temp;
    if (templ1.channels() > 2) {
        cvtColor(templ1, temp, CV_BGR2GRAY);
    }
    else {
        temp = templ1.clone();
    }
    Mat roi2;
    //	Mat img1=inimg.clone();
    //Mat colorimg=inimg.clone();
    //Mat imgbgr;
    //if(img.channels()>2)
    //{
    //	cvtColor(img,imgbgr,CV_BGR2GRAY);
    //
    //}
    //imgbgr=img.clone();

    //Mat imgth;
    //threshold(imgbgr,imgth,100,255,CV_THRESH_BINARY);
    //imwrite("C:/Users/VisionSystem/Desktop/connector/50.jpg",imgth);
    //waitKey(0);
    //	cvtColor(roi,roi,CV_BGR2GRAY);
    int rotationAngles = 90;

    resize(roi, roi2, cv::Size(roi.cols / 2, roi.rows / 2));
    imshow("roi2", roi2);
    Mat temp_roi2;
    resize(temp, temp_roi2, cv::Size(temp.cols / 2, temp.rows / 2));
    imshow("temp_roi2", temp_roi2);
    // imshow("template",temp_roi2);
    //cv::Point2f center1(roi2.cols / 2.0, roi2.rows / 2.0);
    cv::Point2f center1(temp_roi2.cols / 2.0, temp_roi2.rows / 2.0);
    cv::Size a1 = cv::Size(temp_roi2.cols, temp_roi2.rows);
    vector<double> Minvalues(360);
    Mat roi_rot;
    int result_cols = roi2.cols - temp_roi2.cols + 1;
    int result_rows = roi2.rows - temp_roi2.rows + 1;
    Mat dstImage;
    dstImage.create(result_rows, result_cols, CV_32FC1);
    int match_method = 0; //0-4
    double matchTemp;
    double minVal = 1000000000000;
    double angleBest = 0;
    double minValBest = 1000000000000;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    double angleInc = 0;
    float indexer;
    for (int i = -1 * 2 * rotationAngles; i < 2 * rotationAngles; i++) {
        indexer = i;
        if (i == 0)
            angleInc = 0;
        else {
            angleInc = indexer / 2;
        }
        /*if (i>0)
		{int rem =i%2;
		if (rem ==1)
		angleInc=(i/2)+0.5;
		else
		angleInc=i/2;
		}
		else
		{    int temp=-1*i;
		int rem = temp%2;
		if (rem ==1)
		angleInc=(i/2)-0.5;
		else
		angleInc=i/2;
		}*/
        //	 cout<<"angle:::::::::::"<<angleInc<<endl;
        Mat rot_mat = getRotationMatrix2D(center1, double(angleInc), 1.0);
        // warpAffine(edgeImage2, edgeImage2_rot, rot_mat, a, 1);
        //warpAffine(roi2, roi_rot, rot_mat, a1, 1);
        Mat templRot;
        warpAffine(temp_roi2, templRot, rot_mat, a1, 1);
        //	imshow("temp_roi2",temp_roi2);
        //---------------------------------

        cv::warpAffine(temp_roi2, templRot, rot_mat, cv::Size(temp_roi2.cols, temp_roi2.rows));

        //		imshow("templRot",templRot);
        //waitKey(5);
        cv::Point matchLoc;
        matchTemplate(roi2, templRot, dstImage, match_method);
        minMaxLoc(dstImage, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
        //Minvalues[i] = minVal;
        /*rectangle(roi2, minLoc, cv::Point(minLoc.x + temp_roi2.cols, minLoc.y + temp_roi2.rows), Scalar(0, 0, 0), 2, 8, 0);
		imshow("rotImg",roi_rot);
		waitKey(0);*/
        if (minVal < minValBest) {
            minValBest = minVal;

            angleBest = angleInc;
        }
        //if (i==0)
        //	{matchTemp=minVal;
        //cout << "min_temp_value: "<< minVal <<endl;}
        //else
        //{if (minVal<matchTemp)
        //{
        //	matchTemp=minVal;
        //cout << "min_temp_value: "<< minVal <<"i="<<i<<endl;
        //	imshow("roi_rot",roi_rot);
        //}
        //}

        // waitKey();
        // waitKey(0);
        //cout << "min_temp_value: "<< minVal2 << "\t" << i<<endl;
    }
    //cout<<"best angle:::::::"<<angleBest<<endl;
    double min_temp_value = minValBest;
    double angle = -1 * angleBest;
    //*angleRet = angleBest;
    //	 cout<<"Angle"<<angleBest<<endl;
    //	 cout<<"Value"<<minValBest<<endl;
    //for (int i = 0; i < rotationAngles ; i++ )
    //{
    // if(Minvalues[i]  < min_temp_value)
    // {
    //	 min_temp_value = Minvalues[i] ;
    //	 angle = i;
    // }
    //}
    //  cout << "angle: " << angle <<endl;
    // cout << "min_temp_value: "<< min_temp_value <<endl;
    cv::Point2f center2(roi.cols / 2.0, roi.rows / 2.0);
    cv::Size a2 = cv::Size(roi.cols, roi.rows);
    cv::Size a3 = cv::Size(grayImgToBeRotated.cols, grayImgToBeRotated.rows);
    Mat rot_mat = getRotationMatrix2D(center2, angle, 1.0);

    warpAffine(roi, roi_rot, rot_mat, a2, 1);
    Mat returnImage;

    warpAffine(grayImgToBeRotated, returnImage, rot_mat, a3, 1);
    //warpAffine(grayImgToBeRotated2, grayImgToBeRotated2, rot_mat, a3, 1);
    matchTemplate(roi_rot, temp, dstImage, match_method);
    minMaxLoc(dstImage, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    //*matchCoord = minLoc;
    //*matchValue = minVal;
    //	 rectangle( returnImage, minLoc, Point( minLoc.x + temp.cols , minLoc.y + temp.rows ), Scalar::all(0), 2, 8, 0 );
    //
    /* imshow("returnImage",returnImage);
	 waitKey(0);*/
	cout<<"run"<<endl;
    Rect r2 = Rect(Point(minLoc.x - 700, minLoc.y - 50), Point(minLoc.x + 1000, minLoc.y + 1700));

    // Rect r2=Rect( minLoc, Point( minLoc.x + temp.cols , minLoc.y + temp.rows ));
    Mat crp = returnImage(r2);
    //	 imshow("rturn img",returnImage);
    //	  imwrite("C:/Users/VisionSystem/Desktop/connector/111.jpg",crp);
    // imshow("crpcrp img",crp);
    //  waitKey(0);
    //	 imshow("rot1 img",rot1);
    /*imshow("returnImage img",returnImage);
	 waitKey(0);*/
    return crp;
}
Mat resz;
Mat temp_match_with_rotation(Mat inimg, Mat templ1)
{

    Mat grayImgToBeRotated = inimg.clone();
    Mat img1 = inimg.clone();
    //		imshow("thimg11111",img);

    Mat img;

    if (img1.channels() > 2) {
        cvtColor(img1, img, CV_BGR2GRAY);
    }
    else {
        img = img1.clone();
    }
    Mat imgth1;

    threshold(img, imgth1, 240, 255, THRESH_BINARY);
	resize(imgth1,resz,Size(),0.3,0.3);
//	imshow("imgthggg1",resz);
    Mat roi = imgth1.clone();

    //cout<<"chnl+++++++++"<<roi.channels()<<endl;
    //cout<<"chnl:::::::"<<temp.channels()<<endl;
    Mat temp;
    if (templ1.channels() > 2) {
        cvtColor(templ1, temp, CV_BGR2GRAY);
    }
    else {
        temp = templ1.clone();
    }
    Mat roi2;
    //	Mat img1=inimg.clone();
    //Mat colorimg=inimg.clone();
    //Mat imgbgr;
    //if(img.channels()>2)
    //{
    //	cvtColor(img,imgbgr,CV_BGR2GRAY);
    //
    //}
    //imgbgr=img.clone();

    //Mat imgth;
    //threshold(imgbgr,imgth,100,255,CV_THRESH_BINARY);
    //imwrite("C:/Users/VisionSystem/Desktop/connector/50.jpg",imgth);
    //waitKey(0);
    //	cvtColor(roi,roi,CV_BGR2GRAY);
    int rotationAngles = 20;

    resize(roi, roi2, cv::Size(roi.cols / 2, roi.rows / 2));
  //  imshow("roi2", roi2);
    Mat temp_roi2;
    resize(temp, temp_roi2, cv::Size(temp.cols / 2, temp.rows / 2));
 //   imshow("temp_roi2", temp_roi2);
    // imshow("template",temp_roi2);
    //cv::Point2f center1(roi2.cols / 2.0, roi2.rows / 2.0);
    cv::Point2f center1(temp_roi2.cols / 2.0, temp_roi2.rows / 2.0);
    cv::Size a1 = cv::Size(temp_roi2.cols, temp_roi2.rows);
    vector<double> Minvalues(360);
    Mat roi_rot;
    int result_cols = roi2.cols - temp_roi2.cols + 1;
    int result_rows = roi2.rows - temp_roi2.rows + 1;
    Mat dstImage;
    dstImage.create(result_rows, result_cols, CV_32FC1);
    int match_method = 0; //0-4
    double matchTemp;
    double minVal = 1000000000000;
    double angleBest = 0;
    double minValBest = 1000000000000;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    double angleInc = 0;
    float indexer;
    for (int i = -1 * 2 * rotationAngles; i < 2 * rotationAngles; i++) {
        indexer = i;
        if (i == 0)
            angleInc = 0;
        else {
            angleInc = indexer / 2;
        }
        /*if (i>0)
		{int rem =i%2;
		if (rem ==1)
		angleInc=(i/2)+0.5;
		else
		angleInc=i/2;
		}
		else
		{    int temp=-1*i;
		int rem = temp%2;
		if (rem ==1)
		angleInc=(i/2)-0.5;
		else
		angleInc=i/2;
		}*/
        //	 cout<<"angle:::::::::::"<<angleInc<<endl;
        Mat rot_mat = getRotationMatrix2D(center1, double(angleInc), 1.0);
        // warpAffine(edgeImage2, edgeImage2_rot, rot_mat, a, 1);
        //warpAffine(roi2, roi_rot, rot_mat, a1, 1);
        Mat templRot;
        warpAffine(temp_roi2, templRot, rot_mat, a1, 1);
        //	imshow("temp_roi2",temp_roi2);
        //---------------------------------

        cv::warpAffine(temp_roi2, templRot, rot_mat, cv::Size(temp_roi2.cols, temp_roi2.rows));

        //		imshow("templRot",templRot);
        //waitKey(5);
        cv::Point matchLoc;
        matchTemplate(roi2, templRot, dstImage, match_method);
        minMaxLoc(dstImage, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
        //Minvalues[i] = minVal;
        /*rectangle(roi2, minLoc, cv::Point(minLoc.x + temp_roi2.cols, minLoc.y + temp_roi2.rows), Scalar(0, 0, 0), 2, 8, 0);
		imshow("rotImg",roi_rot);
		waitKey(0);*/
        if (minVal < minValBest) {
            minValBest = minVal;

            angleBest = angleInc;
        }
        //if (i==0)
        //	{matchTemp=minVal;
        //cout << "min_temp_value: "<< minVal <<endl;}
        //else
        //{if (minVal<matchTemp)
        //{
        //	matchTemp=minVal;
        //cout << "min_temp_value: "<< minVal <<"i="<<i<<endl;
        //	imshow("roi_rot",roi_rot);
        //}
        //}

        // waitKey();
        // waitKey(0);
        //cout << "min_temp_value: "<< minVal2 << "\t" << i<<endl;
    }
    //cout<<"best angle:::::::"<<angleBest<<endl;
    double min_temp_value = minValBest;
    double angle = -1 * angleBest;
    //*angleRet = angleBest;
    //	 cout<<"Angle"<<angleBest<<endl;
    //	 cout<<"Value"<<minValBest<<endl;
    //for (int i = 0; i < rotationAngles ; i++ )
    //{
    // if(Minvalues[i]  < min_temp_value)
    // {
    //	 min_temp_value = Minvalues[i] ;
    //	 angle = i;
    // }
    //}
    //  cout << "angle: " << angle <<endl;
    // cout << "min_temp_value: "<< min_temp_value <<endl;
    cv::Point2f center2(roi.cols / 2.0, roi.rows / 2.0);
    cv::Size a2 = cv::Size(roi.cols, roi.rows);
    cv::Size a3 = cv::Size(grayImgToBeRotated.cols, grayImgToBeRotated.rows);
    Mat rot_mat = getRotationMatrix2D(center2, angle, 1.0);

    warpAffine(roi, roi_rot, rot_mat, a2, 1);
    Mat returnImage;

    warpAffine(grayImgToBeRotated, returnImage, rot_mat, a3, 1);
    //warpAffine(grayImgToBeRotated2, grayImgToBeRotated2, rot_mat, a3, 1);
    matchTemplate(roi_rot, temp, dstImage, match_method);
    minMaxLoc(dstImage, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    //*matchCoord = minLoc;
    //*matchValue = minVal;
    //	 rectangle( returnImage, minLoc, Point( minLoc.x + temp.cols , minLoc.y + temp.rows ), Scalar::all(0), 2, 8, 0 );
    //
    /* imshow("returnImage",returnImage);
	 waitKey(0);*/

  //  Rect r2 = Rect(Point(minLoc.x - 550, minLoc.y - 850), Point(minLoc.x + 750, minLoc.y + 400));
	 Rect r2 = Rect(Point(minLoc.x - 700, minLoc.y - 1100), Point(minLoc.x + 950, minLoc.y + 500));
    // Rect r2=Rect( minLoc, Point( minLoc.x + temp.cols , minLoc.y + temp.rows ));
    Mat crp = returnImage(r2);
    //	 imshow("rturn img",returnImage);
    //	  imwrite("C:/Users/VisionSystem/Desktop/connector/111.jpg",crp);
    // imshow("crpcrp img",crp);
    //  waitKey(0);
    //	 imshow("rot1 img",rot1);
    /*imshow("returnImage img",returnImage);
	 waitKey(0);*/
    return crp;
}

//int pix_val;
//vector<Point>blck_point;
int mov_along_rows_left2right(Point a,Point b,int width,Mat inimg4,Mat thrs_img,Point *strt_point,Point *end_point)
{
	for (int i = a.y; i <( b.y)-1;i=i+1)
	{
		for (int j = (a.x)-width; j < (b.x)+width; j++)
		{
			//cout << "11111====" << endl;

		//	imshow("thrs_img", thrs_img);   
			//
			pix_val = thrs_img.at<uchar>(i, j);
		//	cout << " (a.x)-width:::::::::::" << (a.x) - width << endl;
			//cout << "  a.y:::::::::::" << a.y << endl;
			
			if (pix_val >20)
			{
			//	circle(inimg4,Point(j,i),3,Scalar(255,0,0),-1);
				blck_point.push_back(Point(j,i));
				break;
			}

			/*else
			{
			circle(inimg4,Point(j,i),3,Scalar(0,250,0),-1);
			}*/

			/*resize(inimg4,resz,Size(),0.5,0.5);
			imshow(" clr  img ",resz);
			waitKey(0);*/
		}
	}
	
	int n = blck_point.size();
	cout << blck_point[n - 1] << endl;
	*strt_point = blck_point[0];
	*end_point = blck_point[n - 1];
	int countr=0;
	for(int k=0;k<blck_point.size()-1;k=k+1)
	{
	//if(blck_point[k].x>blck_point[k+1].x)

	//	cout<<"diffff----"<<abs((blck_point[k+1].x-blck_point[k].x) )<<endl;

		int pix_diff=abs((blck_point[k+1].x-blck_point[k].x) );

		if((blck_point[k+1].x<blck_point[k].x) || pix_diff>2  )
	{
	//	line(inimg4,blck_point[k],blck_point[k+1],Scalar(0,250,0),6);
		circle(inimg4,Point(blck_point[k].x,blck_point[k].y),5,Scalar(0,0,250),-1);

	cout<<"sssssssss"<<endl;
	countr++;
	}
	}

	cout<<"countr----------"<<countr<<endl;
	
	if(countr>0)
	{
		putText(inimg4,"CASTING DEFECT",Point(400,100),FONT_HERSHEY_PLAIN,4,Scalar(0,0,255),4,4 );
	
	}
	else
	{
	putText(inimg4,"OK PART",Point(400,100),FONT_HERSHEY_PLAIN,4,Scalar(0,250,0),4,4 );
	}

	//blck_point.clear();
	return 1; 
}
   vector<vector<Point> > contour;

    vector<Vec4i> hierarcy;
 Mat rs;
Mat Metal_casting_defect(Mat inimg)
{
    Mat img = inimg.clone();

    //Mat img_gray;
    //Mat temp=imread("F:/baumer img/metal_part/temp.jpg");

    //cout<<"img chnl-----------"<<img.channels()<<endl;

    //cout<<"temp chnl-----------"<<temp.channels()<<endl;
    //
    //
    //cvtColor(img,img_gray,CV_BGR2GRAY);

    //
    //Mat temp_gray;
    //	cvtColor(temp,temp_gray,CV_BGR2GRAY);
    // matchTemplate( img_gray, temp_gray, result, match_method );

    // double minVal; double maxVal; Point minLoc; Point maxLoc;
    // Point matchLoc;
    //

    // minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

    //
    // if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
    //   { matchLoc = minLoc; }
    // else
    //   { matchLoc = maxLoc; }
    //cout<<"matchLoc::"<<matchLoc<<endl;
    // Rect crp=Rect( Point(matchLoc.x,matchLoc.y),Point(matchLoc.x,matchLoc.y));
    ////  rectangle( inImage, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar(0,255,0), 2, 8, 0 );
    //   Mat crpimg=img(crp);
    //
    //resize(crpimg,resz,Size(),0.5,0.5);
    //imshow("crpimg",resz);
    //waitKey(0);

    Mat crpimg_gr;
    cvtColor(img, crpimg_gr, CV_BGR2GRAY);

    vector<vector<Point> > contour;

    vector<Vec4i> hierarcy;
    Mat crpimg_th;
    threshold(crpimg_gr, crpimg_th, 245, 255, THRESH_BINARY_INV);
	Mat crpimg_th1=crpimg_th.clone();

    resize(crpimg_th, resz, Size(), 0.5, 0.5);
   // imshow("crpimg_th", resz);
//	waitKey(0);
	//	imwrite("F:/baumer img/metal_part/crpimg_th_new.jpg",crpimg_th);
    //imwrite("F:/baumer img/metal_part/crpth.jpg",crpimg_th);

    Point cntr;
    int width1;
    int height1;
    findContours(crpimg_th, contour, hierarcy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
 //   cout << "3333333333333333333333" << endl;
    cout << "contours.size()----------------" << contour.size() << endl;
    // cout<<"sz="<< contours.size()<<endl;
    for (int i = 0; i < contour.size(); i++) {
        double contAr = contourArea(contour[i], false);

        Rect boundrect = boundingRect(contour[i]);
        width1 = boundrect.width;
        height1 = boundrect.height;

        cout << "height_" << i << "==" << height1 << endl;
        cout << "width_" << i << "==" << width1 << endl;

        if (contAr > 10000) {
            cntr = Point(abs(boundrect.x + (boundrect.x + boundrect.width)) / 2, abs(boundrect.y + (boundrect.y + boundrect.height)) / 2);
            drawContours(img, contour, i, Scalar(250, 0, 0), 2, 3, hierarcy, 0, Point());
            circle(img, cntr, 5, Scalar(250, 0, 0), -1);

        //    circle(img, cntr, ((width1 / 4) + (height1 / 4)), Scalar(255, 0, 0), 4);
         //   rectangle(img, boundrect.br(), boundrect.tl(), Scalar(255, 0, 0), 3, 3);
            cout << "centr---------" << cntr << endl;
        }
    }
//	imwrite("F:/baumer img/metal_part/img.bmp",img);
	
	//Mat chck_new=img(Rect(105,129,741,1553));

	/*129,593,285,590
		525,1305,750,285
		1270,693,305,201*/

		

	/*Mat hsvImg;
		cvtColor(chck_new, hsvImg, CV_BGR2HSV);
		Scalar avgColor= sum(hsvImg)/(hsvImg.cols*hsvImg.rows);
		cout << "avgHue::" << avgColor[0] << endl;
		cout << "avgscale::" << avgColor[1] << endl;
		cout << "avgvalue::" << avgColor[2] << endl;

		if(avgColor[2]<62)
	{
		rectangle(img,Rect(129,593,285,590),Scalar(0,0,250),3,3);
	rectangle(img,Rect(525,1305,750,285),Scalar(0,0,250),3,3);
	rectangle(img,Rect(1270,693,305,201),Scalar(0,0,250),3,3);

		
		putText(img,"PLATING DEFECT",Point(400,100),FONT_HERSHEY_PLAIN,4,Scalar(0,0,255),4,4 );
	
	}
	else
	{
		rectangle(img,Rect(129,593,285,590),Scalar(0,255,0),3,3);
	rectangle(img,Rect(525,1305,750,285),Scalar(0,255,0),3,3);
	rectangle(img,Rect(1270,693,305,201),Scalar(0,255,0),3,3);
	putText(img,"OK PART",Point(400,100),FONT_HERSHEY_PLAIN,4,Scalar(0,250,0),4,4 );
	}

		 resize(img,rs,Size(),0.4,0.4);
	imshow("img",rs);
	waitKey(0);*/
    float px;
    float py;
    //	 int pixval = 0;

    // cv::Point2f center = Point(251, 241);
    int radius = 680;
    bool flag11 = false;
    std::vector<int> blackpixl;
    std::vector<int> whitepixl2;
    int pixval;
    // circle(img_boundry_circle_outer_clr_img, img_centre, 360, Scalar(255, 0, 0), 2, 3);
    // circle(img_boundry_circle_outer_clr_img, img_centre, 300, Scalar(0, 250, 0), 2, 1);

    // circle(img_boundry_circle_outer, img_centre, 30, cv::Scalar(0, 255, 0));

    // circle(img_boundry_circle_outer, img_centre, radius, Scalar(0, 250, 25), 3, 8, 0);
    int counter2;
    for (int degree = 1; degree <= 361; degree = degree + 1) {
        float radian = (degree / (180 / 3.14));
        float x1 = cntr.x + (radius * cos(radian));
        float y1 = cntr.y + (radius * sin(radian));

      //  	 line(img, Point(x1, y1), cntr, Scalar(255, 0, 0), 2, 5, 0);

        /* imshow("bfgchgjkm",img);
				 waitKey(0); */

        double vx = cntr.x - x1; // x vector
        double vy = cntr.y - y1; // y vector

        double mag = sqrt(vx * vx + vy * vy); // length

        vx /= mag;
        vy /= mag;
        int counter = 0;
        // calculate the new vector, which is x2y2 + vxvy * (  distance).
        counter2 = 0;

        for (double distance = 0; distance < radius - 620; distance = distance + 2) {

            px = (int)((float)x1 + vx * ((distance)));

            py = (int)((float)y1 + vy * ((distance)));
            // cout<<"p[x,y]==["<<px<<","<<py<<"]"<<endl;

            //waitKey(0);
            pixval = crpimg_th.at<uchar>(py, px);
            //	  cout << "pixval:::" << pixval << endl;
            if (pixval > 0) {
                circle(img, Point(px, py), 1, Scalar(0, 0, 250), 2, 2, 0);
                counter2++;
            }

            /*else
			 {
			 circle(img_boundry_circle_outer_clr_img, Point(px, py), 1, Scalar(0, 255, 0), 2, 2, 0);
			 }*/
        }
        //	 cout << "counter2222:::::::" << counter2 << endl;
        if (counter2 > 0) {
            whitepixl2.push_back(counter2);
        }
    }
   
   // waitKey(0);

    // cout << "time taken::::::::" << double(clock() - tStart) / CLOCKS_PER_SEC << endl;
    cout << "whitepixl22222:::::::" << whitepixl2.size() << endl;

   /* resize(img, resz, Size(), 0.5, 0.5);
    imshow("imgdfd", resz);*/
    

//	line(crpimg_th1,Point(175,791),Point(288,984),Scalar(0,0,0),4);

	/*Mat crp1 = crpimg_th1(Rect(158, 800, 130, 165));
	crp1=255-crp1;*/
//	 resize(crpimg_th1, resz, Size(), 0.5, 0.5);
//	imshow("crcrpimg_th1p1",resz);
	Point strt;
	Point end;
	
	//mov_along_rows_left2right(Point(175,791), Point(288,984), 20, img, crpimg_th1, &strt, &end);//imageP=color img,threshCrop=threshold img
	mov_along_rows_left2right(Point(239,940), Point(327,1122), 20, img, crpimg_th1, &strt, &end);//imageP=color img,threshCrop=threshold img
	//cout<<"blk point ------------"<<blck_point<<endl;

	// resize(img, resz, Size(), 0.5, 0.5);
  //  imshow("bfgchjkhhgjkm", resz);
  //  waitKey(0);

    return (img);
}
	Mat img_prcss;
bool  algorithmLib::Class1::jaquarM22(int maskEn, System::Drawing::Bitmap ^ bitmap0)
{
    //string Path="D:/CV/08 august/New folder (2)/circle_ng/";
    //string Path="D:/CV/08 august/New folder (2)/circle_ng/";
    out = 0;
    // try {
    clock_t tStart = clock();
    /*	imwrite("1.bmp",BitmapToMat (bitmap0));
Mat img=imread("1.bmp");*/
	Mat rs;
     img_prcss = BitmapToMat(bitmap0);
    //Mat bgr[3];
    //split(img,bgr);
    //if (maskEn==0)
    // red=bgr[0].clone();
    //else if (maskEn==1)
    //red=bgr[1].clone();
    //else
    //red=bgr[2].clone();
    ////imshow("red",bgr[2]);
    ////waitKey(0);
    //cvtColor(red,red,CV_GRAY2BGR);


	//---------------------------------------------

	 Mat temp = imread("F:/baumer img/metal_part/crp1.jpg");
	Mat temp1;
	cvtColor(temp,temp1, CV_BGR2GRAY);

	threshold(temp1, temp1,240,255,THRESH_BINARY);
	//imshow("temp1",temp1);
		



	

		 Mat rotated_img1 = temp_match_with_rotation(img_prcss, temp1);
		
		 resize(rotated_img1,rs,Size(),0.3,0.3);
		// imshow("rotated_img1",rs);
	//	 	waitKey(0);
//    Mat rotated_img = temp_match_with_rotation(img, temp);
//
////	imwrite("F:/baumer img/metal_part/rotated_img.jpg",rotated_img);
//
//    resize(rotated_img, resz, Size(), 0.5, 0.5);
//    imshow("rotated_img", resz);
	int result;
 Mat outimg=   Metal_casting_defect(rotated_img1);
 resize(outimg,img_prcss,img_prcss.size());
		// imshow("outimg",rs);
		
		// rs.copyTo(img_prcss);
		 // resize(img_prcss,resz,Size(),0.5,0.5);
		// imshow("img_prcss",img_prcss);
		// waitKey(0);
		// imwrite("F:/data1/1.jpg",img_prcss);
		 
	//----------------------------------------------------------------------------------------------------

		 



 //   Mat temp = imread("F:/baumer img/key/temp.jpg");
 //   resize(temp, rz, Size(), 0.3, 0.3);
 // //  imshow("temp", rz);
 //   //  waitKey(0);
 //   int rot = 1;
 //   Mat img_rot = img.clone();
 //   Mat rotated = Template_match_with_rotation_match0(img, temp, img_rot, rot);

	//int cn;
 //   Mat outimg = similarity_chck(rotated,&cn);
	//Mat rz;
	//resize(outimg,rz,img.size());
	//rz.copyTo(img);

	bool rslt;
	//if(cn>700)
	//{
	//	putText(img, "Key Missmatch",Point(300,100),FONT_HERSHEY_DUPLEX,2,Scalar(0,0,255),2);
	//rslt=0;
	//}
	//else
	//{
	//putText(img, "Key match",Point(300,100),FONT_HERSHEY_DUPLEX,2,Scalar(0,250,0),2);
	//rslt=1;
	//}



//	imshow("img",img);
//	waitKey(0);
    cout << "time taken::::::::" << double(clock() - tStart) / CLOCKS_PER_SEC;
    //------------------------------------------------------------------------------

    //-------------------------------------------------------------------------
    //cvtColor(imgRet,imgRet,CV_GRAY2BGR);
   // System::Drawing::Bitmap ^ dst = MatToBitmap(img);
    //return dst;
    //-----------------------------------------------------------------------

    //return dst;
    ////}
    ////catch( exception ex)
    ////{
    //return bitmap0;
    ////}
	return rslt;
}

	
bool  algorithmLib::Class1::jaquar(int maskEn, System::Drawing::Bitmap ^ bitmap0)
{
    //string Path="D:/CV/08 august/New folder (2)/circle_ng/";
    //string Path="D:/CV/08 august/New folder (2)/circle_ng/";
    out = 0;
    // try {
    clock_t tStart = clock();
    /*	imwrite("1.bmp",BitmapToMat (bitmap0));
Mat img=imread("1.bmp");*/

    Mat img = BitmapToMat(bitmap0);
    //Mat bgr[3];
    //split(img,bgr);
    //if (maskEn==0)
    // red=bgr[0].clone();
    //else if (maskEn==1)
    //red=bgr[1].clone();
    //else
    //red=bgr[2].clone();
    ////imshow("red",bgr[2]);
    ////waitKey(0);
    //cvtColor(red,red,CV_GRAY2BGR);

    Mat temp = imread("F:/baumer img/key/temp.jpg");
    resize(temp, rz, Size(), 0.3, 0.3);
  //  imshow("temp", rz);
    //  waitKey(0);
    int rot = 1;
    Mat img_rot = img.clone();
    Mat rotated = Template_match_with_rotation_match0(img, temp, img_rot, rot);

	int cn;
    Mat outimg = similarity_chck3(rotated,&cn);
	Mat rz;
	resize(outimg,rz,img.size());
	rz.copyTo(img);

	bool rslt;
	if(cn>2500)
	{
		putText(img, "Key Missmatch",Point(300,100),FONT_HERSHEY_DUPLEX,2,Scalar(0,0,255),2);
	rslt=0;
	}
	else
	{
	putText(img, "Key match",Point(300,100),FONT_HERSHEY_DUPLEX,2,Scalar(0,250,0),2);
	rslt=1;
	}


	
//	imshow("img",img);
//	waitKey(0);
    cout << "time taken::::::::" << double(clock() - tStart) / CLOCKS_PER_SEC;
    //------------------------------------------------------------------------------

    //-------------------------------------------------------------------------
    //cvtColor(imgRet,imgRet,CV_GRAY2BGR);
   // System::Drawing::Bitmap ^ dst = MatToBitmap(img);
    //return dst;
    //-----------------------------------------------------------------------

    //return dst;
    ////}
    ////catch( exception ex)
    ////{
    //return bitmap0;
    ////}
	return rslt;
}
