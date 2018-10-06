#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <memory>

using namespace cv;
using namespace std;

const GLdouble size=0.40;

class AR_GL_CV_Tester
{
public:
  AR_GL_CV_Tester();
  AR_GL_CV_Tester(const AR_GL_CV_Tester& other);
  ~AR_GL_CV_Tester();

  bool initGL(); //setup GL for drawing // This is in the master branch
  bool initCV(); //setup CV
  void display(); //our callback function
  void drawQuad(Mat&, Mat, Scalar);
  void handleMarkerTracking();
  void captureProcessWebcam();
  const Mat& getImg() const {return m_img;}


private:
  Mat m_img;
  Mat m_rvec;
  Mat m_tvec;
  Mat m_intrinsics;
  Mat m_distortion;
  Mat m_glViewMatrix;
  VideoCapture m_capture;
  vector<vector<Point> > m_contours;
  vector<Mat> m_squares;
	GLuint textureID;
  GLuint textureID2;

};

AR_GL_CV_Tester::AR_GL_CV_Tester()
{

}

AR_GL_CV_Tester::AR_GL_CV_Tester(const AR_GL_CV_Tester& other)
{

}

AR_GL_CV_Tester::~AR_GL_CV_Tester()
{

}


void AR_GL_CV_Tester::drawQuad(Mat& image, Mat points, Scalar color) {
    cout << points.at<Point2f>(0,0) << " " << points.at<Point2f>(0,1) << " " << points.at<Point2f>(0,2) << " " << points.at<Point2f>(0,3) << endl;
    line(image, points.at<Point2f>(0,0), points.at<Point2f>(0,1), color);
    line(image, points.at<Point2f>(0,1), points.at<Point2f>(0,2), color);
    line(image, points.at<Point2f>(0,2), points.at<Point2f>(0,3), color);
    line(image, points.at<Point2f>(0,3), points.at<Point2f>(0,0), color);
}

//INIT TO GL
bool AR_GL_CV_Tester::initGL(void)
{
    //select clearing (background) color
    glClearColor(1.0, 1.0, 1.0, 1.0);
    //glClearDepth(1.0);
    //glDepthFunc(GL_LESS);
    //glEnable(GL_DEPTH_TEST);
    //glShadeModel(GL_SMOOTH);
    //glMatrixMode(GL_PROJECTION)

    //initialize viewing values
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
    auto imageWidth = 640;
    auto imageHeight = 480;
    auto fx = m_intrinsics.at<double>(0,0);
    auto fy = m_intrinsics.at<double>(1,1);
    auto centerImageX = m_intrinsics.at<double>(0,2);
    auto centerImageY = m_intrinsics.at<double>(1,2);

    float near = 0.1;
    float far = 100.0;

    /*
    glFrustum(-near*centerImageX/(GLfloat)fx,
              near*(imageWidth-centerImageX)/(GLfloat)fx,
              near*(centerImageY-imageHeight)/(GLfloat)fy,
              near*centerImageY/(GLfloat)fy, near, far);
    */

    auto fovy = 2*atan(0.5*imageHeight/fy)*180/M_PI;
    auto aspect = (imageWidth*fy)/(imageHeight*fx);

    gluPerspective(fovy,aspect,near,far);
    glViewport(0,0,imageWidth,imageHeight);




    //gluPerspective(45.0f, 500.0f / 500.0f, 0.1f, 100.0f);

    //glMatrixMode(GL_MODELVIEW);


    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &textureID); //videobackground
    glGenTextures(1, &textureID2); //teapot texture


    return true;
}

bool AR_GL_CV_Tester::initCV()
{
  //read in the camera calibration
  FileStorage fs("../../calibrate/build/out_camera_data.xml", FileStorage::READ);
  fs["Camera_Matrix"] >> m_intrinsics;
  fs["Distortion_Coefficients"] >> m_distortion;
  if (m_intrinsics.rows != 3 || m_intrinsics.cols != 3 || m_distortion.rows != 5 || m_distortion.cols != 1) {
      cout << "Run calibration (in ../calibrate/) first!" << endl;
      return false;
  }
  cout << "intrinsics " << m_intrinsics << endl;
  //open the camera
  bool result = m_capture.open(-1);
  if (! result){
    return false;
  }
  return true;
}

//do all the OCV processing here
void AR_GL_CV_Tester::handleMarkerTracking()
{
  Scalar green(0, 255, 0);
  //clear containers!!!!!!!!!!!!!!
  m_contours.clear();
  m_squares.clear();
  //convert to greyscale
  Mat grayImage;
  cvtColor(m_img, grayImage, CV_RGB2GRAY);
  //blur
  Mat blurredImage;
  blur(grayImage, blurredImage, Size(5, 5));
  //threshold
  Mat threshImage;
  threshold(blurredImage, threshImage, 128.0, 255.0, THRESH_OTSU);
  //get the contours
  findContours(threshImage, m_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

  for (auto contour : m_contours) {
      vector<Point> approx;
      approxPolyDP(contour, approx, arcLength(Mat(contour), true)*0.02, true);
      if( approx.size() == 4 &&
          fabs(contourArea(Mat(approx))) > 1000 &&
          isContourConvex(Mat(approx)) )
      {
          Mat squareMat;
          Mat(approx).convertTo(squareMat, CV_32FC3);
          m_squares.push_back(squareMat);
      }
  }

  if (m_squares.size() > 0) {
    vector<Point3f> objectPoints = {Point3f(-1, -1, 0), Point3f(-1, 1, 0), Point3f(1, 1, 0), Point3f(1, -1, 0)};
    Mat objectPointsMat(objectPoints);
    cout << "objectPointsMat: " << objectPointsMat.rows << ", " << objectPointsMat.cols << endl;
    cout << "squares[0]: " << m_squares[0] << endl;
    solvePnP(objectPointsMat, m_squares[0], m_intrinsics, m_distortion, m_rvec, m_tvec);

    cout << "rvec = " << m_rvec << endl;
    cout << "tvec = " << m_tvec << endl;

    //create matrices for OpenGL
    Mat rotation;
    Mat viewMatrix = Mat::zeros(4, 4, CV_64F);
    Rodrigues(m_rvec, rotation);
    for(unsigned int row=0; row<3; ++row)
    {
      for(unsigned int col=0; col<3; ++col)
      {
        viewMatrix.at<double>(row, col) = rotation.at<double>(row, col);
      }
      viewMatrix.at<double>(row, 3) = m_tvec.at<double>(row, 0);
    }

    viewMatrix.at<double>(3, 3) = 1.0f;
    m_glViewMatrix = Mat::zeros(4, 4, CV_64F);
    transpose(viewMatrix , m_glViewMatrix);
    //comp our green poly on the Img
    drawQuad(m_img, m_squares[0], green);
  }



}

void AR_GL_CV_Tester::captureProcessWebcam()
{

  if (m_capture.read(m_img))
  {
    //cout << "read OK" << endl;
  }
  else
  {
    cout << "bad webcam read eh";
  }
}

//OUR DISPLAY CALLBACK FUNCTION
void AR_GL_CV_Tester::display(void)
{


    //Clear all pixels
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();

    //glPushMatrix();
    //draw BKGRND
    Mat flipped;
    transpose(m_img, flipped);
    flip(flipped, flipped, 1);
    //take the video and create BKGRND texture for OpenGL
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR); // scale linearly when image bigger than texture
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); // scale linearly when image smalled than texture
    glTexImage2D(GL_TEXTURE_2D, 0, 3, flipped.cols, flipped.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, flipped.data);

    glBindTexture(GL_TEXTURE_2D, textureID);
    glPushMatrix();
    //glTranslatef(0.0,0.0,-10.0);

    glBegin(GL_QUADS);

    float x = 1;
    float y = 1;

    glTexCoord2f(0.0f, 0.0f); glVertex3f(-x, -y, 0.0f);
    glTexCoord2f(0.0f, 1.0f);glVertex3f( x, -y, 0.0f);
    glTexCoord2f(1.0f, 1.0f);glVertex3f( x, y, 0.0f);
    glTexCoord2f(1.0f, 0.0f);glVertex3f(-x, y, 0.0f);

    glEnd();
    glPopMatrix();


    glBindTexture(GL_TEXTURE_2D, textureID2);
    glMatrixMode(GL_MODELVIEW);
        glRotatef(45,1,0,0);
    glPushMatrix();
    glLoadMatrixd(&m_glViewMatrix.at<double>(0, 0));

    glutSolidTeapot(size);

    glPopMatrix();
    // Don't wait start processing buffered OpenGL routines
    glFlush();

}

//global ptr and display callback wrapper
AR_GL_CV_Tester* g_ARGL;
void oglDraw()
{
  if (g_ARGL)
  {
    g_ARGL->display();
  }
}

void oglIdle()
{
  if (g_ARGL)
  {
    g_ARGL->captureProcessWebcam();
    g_ARGL->handleMarkerTracking();
    glutPostRedisplay();
  }
}

int main( int argc, char** argv ){

  //VideoCapture cap(-1);
  //Mat img;
  AR_GL_CV_Tester ar;
  g_ARGL = &ar;
  bool result;

  result = ar.initCV();
  if (!result)
  {
    return -1;
  }


  glutInit(&argc, argv);
  //set display mode
  glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
  //Set the window size
  glutInitWindowSize(1280,720);
  //Set the window position
  glutInitWindowPosition(100,100);
  //Create the window
  glutCreateWindow("A Simple OpenGL Windows Application with GLUT");

  result = ar.initGL();
  if (!result)
  {
    return -1;
  }

  glutDisplayFunc(oglDraw);
  glutIdleFunc(oglIdle);
  glutMainLoop();


  return 0;
}
