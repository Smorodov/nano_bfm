// http://rodolphe-vaillant.fr/?e=29

#define GL_GLEXT_PROTOTYPES
#include <windows.h>		

#include <time.h>
#include "stdlib.h"
#include <crtdbg.h>
#include <conio.h>
#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#include "GL/glew.h"
#include <GL/gl.h>
#include <GL/glut.h>
#include <GL/freeglut_ext.h>
#include <stdio.h>
#include <cmath>
#include "cnpy.h"
#include "bfm.h"
#include "glm/glm.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/dual_quaternion.hpp>
#include <set>
#include <vector>

int tex_size = 1024;

using namespace std;


// Create shader
char const* vsrc = "\
    #version 330 core                                                           \n\
    layout(location = 0) in vec3 position;                                      \n\
    layout(location = 1) in vec3 normal;                                        \n\
    layout(location = 2) in vec3 color;                                        \n\
    out vec3 data_normal;                                                               \n\
    out vec3 data_position;                                                               \n\
    out vec3 rgb;\n\
    out mat4 data_mvp;\n\
    uniform mat4 mvp;                                              \n\
    void main() {                                                               \n\
      data_position=vec3(mvp * vec4(position, 1.0));                                      \n\
      gl_Position=vec4(data_position,1.0);                                                \n\
      data_mvp=mvp;\n\
      data_normal = (vec4(normal, 1.0)).xyz;                              \n\
      rgb=vec3(color);\n\
    }                                                                           \n\
  ";

char const* fsrc = "\
    #version 330                                                              \n\
    in vec3 data_normal; \n\
    in vec3 data_position; \n\
    in vec3 rgb;\n\
    in mat4 data_mvp; \n\
    out vec4 outputF;                                                             \n\
    void main()                                                                 \n\
    { \n\
    //vec3 rgb = vec3(0.8,0.8,0.8);                                               \n\
    float ambient = 0.2;\n\
    mat4 mvp=transpose(data_mvp); \n\
    vec3 L=normalize(vec3(mvp*vec4(0,0,-1,0)));\n\
    vec3 eye=normalize(vec3(0,0,-1));\n\                                                \n\
    vec3 E = normalize(eye - data_position);\n\
    vec3 H = normalize(L + E);\n\
    vec3 N = data_normal;\n\
	float diffuse = abs(dot(N, L));\n\
    float specular = pow(max(abs(dot(N, H)), 0.0), 12.0);\n\
    float NE = max(abs(dot(N, E)), 0.0);\n\
    float fresnel = pow(sqrt(1.0 - NE * NE), 12.0);\n\
    //outputF = vec4(rgb*0.005, 1.0);\n\
    outputF = vec4((0.5 * diffuse + 0.4 * specular + 0.6 * fresnel + ambient) * rgb*(1.0/255.0), 1.0);\n\
    }\n\
  ";



class MyTrackBall
{
public:
    // Высота и ширина окна отображения
    int width;
    int height;
    int	gMouseX;
    int	gMouseY;
    glm::vec2 lastPos;
    double	Vvx, Vvy, Vvz;
    double	Vglx, Vgly, Vglz;
    double	Vprx, Vpry, Vprz;
    double	mx0, my0, mx1, my1;
    double  vx1, vy1, vz1;
    double	ViewMatrix[16];
    double	xp, yp, zp;
    double	dx, dy, dz;
    double	rx, ry, rz;
    double	kx;
    int		button;
    double	obj_size;
    double	X0, Y0, Z0;
    double	Pi;
    double	deg;
    double	last_ang;
    double	X, Y, Z;
    double	xRot;
    double	yRot;
    double	zRot;

    MyTrackBall(int w, int h, float obj_size = 1.0)
    {
        width = w;
        height = h;
        gMouseX = 0;
        gMouseY = 0;
        button = 0;
        lastPos = glm::vec2(0.0f, 0.0f);
        this->obj_size = obj_size;
        X0 = 0, Y0 = 0, Z0 = 0;
        Pi = 3.14159265358979323846;
        deg = Pi / 180;
        last_ang = 0;
        X = 0, Y = 0, Z = 0;

        xRot = 0;
        yRot = 0;
        zRot = 0;
        dx = 0; dy = 0; dz = 0;
        xp = 0; yp = 0; zp = 0;
        rx = 0; ry = 0; rz = 0;
        mx0 = 0; my0 = 0;
        mx1 = 0; my1 = 0;
        button = 0;
        Vvx = 0; Vvy = 1; Vvz = 0;
        Vglx = 0; Vgly = 0; Vglz = 1;
        Vprx = 1; Vpry = 0; Vprz = 0;
        if (width < height) { kx = 0.5 * obj_size / height; }
        else { kx = 0.5 * obj_size / width; }
    }

    //--------------------------------------------------------------------------
    //
    //--------------------------------------------------------------------------
    void MouseWheelCallback(int direction)
    {
        if (direction > 0)
        {
            obj_size *= 0.95;
        }
        else if (direction < 0)
        {
            obj_size *= 1.05;
        }
    }
    //--------------------------------------------------------------------------  
    // Функция обработки нажатий кнопок мыши
    // mouse left = 1
    // mouse right = 2
    // mouse middle = 3
    //--------------------------------------------------------------------------
    void MouseButtonCallback1(int button, int x, int y)
    {
        this->button = button;
        dx = 0;
        dy = 0;
        lastPos.x = x;
        lastPos.y = y;
    }

    //--------------------------------------------------------------------------
    // Функция обработки перемещений мыши
    //-------------------------------------------------------------------------- 
    void MouseMotionCallback(int x, int y)
    {
        if (button > 0)
        {
            dx = x - lastPos.x;
            dy = y - lastPos.y;
            lastPos.x = x;
            lastPos.y = y;
        }
        else
        {
            dx = 0;
            dy = 0;
        }
    }

    glm::mat4 getMVP(void)
    {
        double nRange = obj_size;
        int w = width;
        int h = height;
        if (h == 0) { h = 1; }
        glm::vec4 viewport = glm::vec4(0, 0, w, h);
        glm::mat4 projection;
        if (w <= h)
        {
            projection = glm::ortho(-nRange, nRange, -nRange * h / w, nRange * h / w, -200 * nRange, 200 * nRange);
        }
        else
        {
            projection = glm::ortho(-nRange * w / h, nRange * w / h, -nRange, nRange, -200 * nRange, 200 * nRange);
        }
        glm::mat4 view(1.0f); // Identity matrix         
        double wx1, wy1, wz1;
        glm::vec3 projected = glm::project(glm::vec3(xp, yp, zp), view, projection, viewport);
        vx1 = projected.x;
        vy1 = projected.y;
        vz1 = projected.z;
        //gluProject(xp, yp, zp, &modelview1[0], &projection1[0], &viewport1[0], &vx1, &vy1, &vz1);

        if (button == 3)
        {
            vx1 = vx1 + dx;
            vy1 = vy1 - dy;
        }
        glm::vec3 unprojected = glm::unProject(glm::vec3(vx1, vy1, 0), view, projection, viewport);
        wx1 = unprojected.x;
        wy1 = unprojected.y;
        wz1 = unprojected.z;
        //gluUnProject(vx1, vy1, 0, modelview1, projection1, viewport1, &wx1, &wy1, &wz1);
        if (button == 3 && wx1 != 0 && wy1 != 0)
        {
            xp = wx1;
            yp = wy1;
            zp = 0;
        }
        view = glm::translate(view, glm::vec3(xp, yp, zp));

        if (button == 1)
        {
            double LVgl = 1, LVv = 1, LVpr = 1;
            dx /= 100;
            Vprx = Vprx - Vglx * (-dx);
            Vpry = Vpry - Vgly * (-dx);
            Vprz = Vprz - Vglz * (-dx);
            dy /= 100;
            Vvx = Vvx - Vglx * (dy);
            Vvy = Vvy - Vgly * (dy);
            Vvz = Vvz - Vglz * (dy);

            Vglx = Vpry * Vvz - Vprz * Vvy;
            Vgly = Vprz * Vvx - Vprx * Vvz;
            Vglz = Vprx * Vvy - Vpry * Vvx;

            Vprx = Vvy * Vglz - Vvz * Vgly;
            Vpry = Vvz * Vglx - Vvx * Vglz;
            Vprz = Vvx * Vgly - Vvy * Vglx;

            LVgl = sqrt(Vglx * Vglx + Vgly * Vgly + Vglz * Vglz);
            LVv = sqrt(Vvx * Vvx + Vvy * Vvy + Vvz * Vvz);
            LVpr = sqrt(Vprx * Vprx + Vpry * Vpry + Vprz * Vprz);

            if (LVgl != 0)
            {
                Vglx = Vglx / LVgl;
                Vgly = Vgly / LVgl;
                Vglz = Vglz / LVgl;
            }
            if (LVpr != 0)
            {
                Vprx = Vprx / LVpr;
                Vpry = Vpry / LVpr;
                Vprz = Vprz / LVpr;
            }
            if (LVv != 0)
            {
                Vvx = Vvx / LVv;
                Vvy = Vvy / LVv;
                Vvz = Vvz / LVv;
            }
        }
        glm::mat4 view_rot;
        if (glm::vec3(Vglx, Vgly, Vglz).length() > 0)
        {
            view_rot = glm::lookAtRH(glm::vec3(Vglx, Vgly, Vglz), glm::vec3(0, 0, 0), glm::vec3(Vvx, Vvy, Vvz));
        }
        else
        {
            view_rot = glm::mat4(1.0f);
        }
        view = view * view_rot;
        // view = glm::translate(view, glm::vec3(X0, Y0, Z0));
         //-------------------------------------------------------------------------       
        dx = 0;
        dy = 0;
        glm::mat4 model(1.0f);
        glm::mat4 MVP = projection * view * model;
        return MVP;
    }

};
MyTrackBall* trackball;
bfm model;

void KeyboardCallback(unsigned char key, int x, int y);
void initialize(void);
void finalize(void);


// Vertex buffer
GLuint vertices = 0;
// Vertex normals
GLuint normals = 0;
// Vertex normals
GLuint colors = 0;
// Vertex indices
GLuint indices = 0;
GLuint program;
GLuint vs;
GLuint fs;

//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------

void RenderString(float xp, float yp, float zp, void* font, const char* string, float r, float g, float b)
{
    glColor3f(r, g, b);
    glRasterPos3f(xp, yp, zp);
    glutBitmapString(font, (unsigned char*)string);
}

//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
float t = 0;
void display(void)
{
    std::vector<float> shape(20,0);
    std::vector<float> tex(20, 0);

    for (int i = 0; i < 20; ++i)
    {
        shape[i] = 3/sqrt(float(i+1))*sin(t * 0.3 * (i+1));
        tex[i] = 3 / sqrt(float(i + 1)) *sin(t * 0.3 * (i+1));
    }
    t += 0.05;

    model.updateMesh(shape,tex);
    float* vertices_data = model.Shape.data<float>();
    float* vetexNormals = model.vetexNormals.data<float>();
   model.normalsComputer->getVertexNormals(model.vertex_num, vertices_data, vetexNormals);              
   float* vetexColors = model.Tex.data<float>();
   glOrtho(-1, 1, -1, 1, -1, 1);
   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   glLoadIdentity();

   glMatrixMode(GL_PROJECTION);
   glPushMatrix();
   glLoadIdentity();
   // Draw mesh

    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1, -1);
    glOrtho(1, -1, 1, -1, -1, 1);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glBindBuffer(GL_ARRAY_BUFFER, vertices);
    glBufferSubData(GL_ARRAY_BUFFER, 0, model.vertex_num * sizeof(float), vertices_data);
    
    glBindBuffer(GL_ARRAY_BUFFER, normals);
    glBufferSubData(GL_ARRAY_BUFFER, 0, model.vertex_num * sizeof(float), vetexNormals);

    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glBufferSubData(GL_ARRAY_BUFFER, 0, model.vertex_num * sizeof(float), vetexColors);

    glBindBuffer(GL_ARRAY_BUFFER, indices);
    glBufferSubData(GL_ARRAY_BUFFER, 0, model.face_index_num * 3 * sizeof(int), (unsigned int*)model.tl.data<int>());
    glUseProgram(program);
    glm::mat4 mvp = trackball->getMVP();
    glUniformMatrix4fv(glGetUniformLocation(program, "mvp"), 1, false, glm::value_ptr(mvp));
    glDrawElements(GL_TRIANGLES, model.face_index_num * 3, GL_UNSIGNED_INT, 0);
    glUseProgram(0);    
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    
    //glOrtho(-1, 1, -1, 1, -1, 1);
    glDisable(GL_LIGHTING);
    
    
   // glColor3f(0.0, 1.0, 0.0);
   // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
   // glDrawElements(GL_TRIANGLES, model.face_index_num * 3, GL_UNSIGNED_INT, 0);
    
    
     // glColor3f(1, 0, 0);
     // glPointSize(2);    
     // glPointSize(2);
     // glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
     // glDrawElements(GL_POINTS, model.face_index_num*3, GL_UNSIGNED_INT, 0);
     // glPointSize(1);  
    
}
//--------------------------------------------------------------------------
// KEYS
//--------------------------------------------------------------------------
static void KeyboardCallback(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 27:
        exit(0);
        break;
    case 'w':
    case 'W':
        break;
    }
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
void MouseWheelCallback(int wheel, int direction, int x, int y)
{
    trackball->MouseWheelCallback(direction);
}
//--------------------------------------------------------------------------  
// Функция обработки нажатий кнопок мыши
//--------------------------------------------------------------------------
static void MouseCallback1(int _button, int state, int x, int y)
{   
    int button = 0;
    if (_button == GLUT_LEFT_BUTTON) { button = 1; }
    if (_button == GLUT_MIDDLE_BUTTON) { button = 3; }
    if (_button == GLUT_RIGHT_BUTTON) { button = 2; }
    trackball->MouseButtonCallback1(button, x, y);
}

//--------------------------------------------------------------------------
// Функция обработки перемещений мыши
//-------------------------------------------------------------------------- 
static void MotionCallback(int x, int y)
{
    trackball->MouseMotionCallback(x, y);
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
static void ArrowKeyCallback(int key, int x, int y)
{
    KeyboardCallback(key, x, y);
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
static void IdleCallback()
{
    glutPostRedisplay();
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
void renderScene(void)
{     
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();            
    
    glm::mat4 mvp=trackball->getMVP();    
    glPushMatrix();
    glLoadMatrixf(glm::value_ptr(mvp));
    // Render all here    
    // Clear buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    display();
    glPopMatrix();    
    //--------------------------------------------------------------------------
    //
    //--------------------------------------------------------------------------
    glFlush();
    glutSwapBuffers();
}
//--------------------------------------------------------------------------
// MAIN
//--------------------------------------------------------------------------
void main(int argc, char** argv)
{
    system("chcp 1251");
    system("cls");
    int width = 800, height = 800;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(width, height);
    trackball = new MyTrackBall(width, height,150000);
    // Create window, set callbacks
    int mainHandle = glutCreateWindow("3D View");
    glutSetWindow(mainHandle);
    glutDisplayFunc(renderScene);
    glutIdleFunc(IdleCallback);
    glutKeyboardFunc(KeyboardCallback);
    glutSpecialFunc(ArrowKeyCallback);
    glutMouseFunc(MouseCallback1);
    glutMotionFunc(MotionCallback);
    glutMouseWheelFunc(MouseWheelCallback);
    glutCloseFunc(finalize);
    MotionCallback(0, 0);

    // Setup default render states
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHTING);
    float ambientColor[] = { 0.5f, 0.5f, 0.5f, 0.0f };
    float diffuseColor[] = { 1.0f, 1.0f, 1.0f, 0.0f };
    float specularColor[] = { 0.5f, 0.5f, 0.5f, 0.0f };
    float position[] = { 100.0f, 100.0f, 400.0f, 1.0f };
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientColor);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseColor);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specularColor);
    glLightfv(GL_LIGHT0, GL_POSITION, position);
    glEnable(GL_LIGHT0);
    initialize();
    glutMainLoop();
}



//--------------------------------------------------------------------------
// START
//--------------------------------------------------------------------------
void initialize(void)
{
    model.loadModel("model/01_MorphableModel.npz");
   
    glewInit();
    glGenBuffers(1, &vertices);
    glGenBuffers(1, &normals);
    glGenBuffers(1, &colors);
    glGenBuffers(1, &indices);
    
    glBindBuffer(GL_ARRAY_BUFFER, vertices);
    glBufferData(GL_ARRAY_BUFFER, model.vertex_num * sizeof(float), NULL, GL_STREAM_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, normals);
    glBufferData(GL_ARRAY_BUFFER, model.vertex_num * sizeof(float), NULL, GL_STREAM_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, false, 0, (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glBufferData(GL_ARRAY_BUFFER, model.vertex_num * sizeof(float), NULL, GL_STREAM_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, false, 0, (void*)0);


    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, model.face_index_num * 3 * sizeof(int), NULL, GL_STREAM_DRAW);

    program = glCreateProgram();
    vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsrc, NULL);
    glCompileShader(vs);
    glAttachShader(program, vs);
    glDeleteShader(vs);
    fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsrc, NULL);
    glCompileShader(fs);
    glAttachShader(program, fs);
    glDeleteShader(fs);
    glLinkProgram(program);
}

//--------------------------------------------------------------------------
// END
//--------------------------------------------------------------------------
void finalize(void)
{
    glDeleteProgram(program);
    glDeleteBuffers(1, &vertices);
    glDeleteBuffers(1, &normals);
    glDeleteBuffers(1, &colors);
    glDeleteBuffers(1, &indices);
    delete trackball;

}





