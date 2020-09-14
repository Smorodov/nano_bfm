// Author: Court Cutting, MD
// Date: May 19, 2012
// Purpose: Example program showing use of glsl and texture buffer objects to
//    do normal computation on the graphics card. Feel free to modify this example
//    for your own purposes. There are no licensing restrictions. It would be kind
//    if you would cite the webpage where the idea originated.

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <vector>
#include <stdio.h>
#include <math.h>

#include <iostream>

GLfloat viewMatrix[16];
float xrot=0.0, yrot=0.0, xIncr=0.0013f, zIncr=0.0033f;	// change speed of vertex movement using xIncr and zIncr
int screenX,screenY,lastX,lastY;
GLfloat tetVerts[] = {-0.5f, 0.5f, 0.0f, 1.0f,
	0.5f, 0.5f, 0.0f, 1.0f,
	0.0f, -0.5f, 0.5f, 1.0f,
	0.0f, -0.5f, -0.5f, 1.0f};

GLuint _NormalMakerProgram=0;
GLuint _RenderingProgram=0;
GLuint _vertexArrayBufferObject=0;
int _nPositions=4,_nVertices=12;
std::vector<GLfloat> _Positions;        // Array of unique positions of objects. For glsl, 4 elements with v[3]=1.0f
GLuint _bufferObjects[3];
GLuint _textureBufferObjects[2];
GLuint _texBOBuffers[2];

static const char *GTNormalMakerVertexShader = "#version 140\n"
	"uniform samplerBuffer positionSampler;\n"
	"uniform usamplerBuffer neighborSampler;\n"
	"out vec3 vSurfaceNormal;\n"
	"out vec4 vSurfaceCoord;\n"
	"void main(void) {\n"
	"	uvec4 nei4 = texelFetch(neighborSampler,gl_VertexID);\n"
	"	vec4 tmp,vertexPosition = texelFetch(positionSampler,int(nei4[0]));\n" // gl_VertexID
	"	vSurfaceCoord = vertexPosition;\n" // mvpMatrix*
	"	vec3 norm,first,second;\n"
	"	tmp = texelFetch(positionSampler,int(nei4[1]));\n"
	"	first = tmp.xyz - vertexPosition.xyz;\n"
	"	tmp = texelFetch(positionSampler,int(nei4[2]));\n"
	"	second = tmp.xyz - vertexPosition.xyz;\n"
	"	norm = cross(first,second);\n"
	"	vSurfaceNormal = normalize(norm);\n"
	"}";

static const char *DumbVertexShader = "#version 140\n"
	"in vec4 vVertex;\n"
	"in vec3 vNormal;\n"
	"in vec4 vColor;\n"
	"uniform mat4   mvpMatrix;\n"
	"smooth out vec3 normal,lightDir;\n"
	"out vec4 vVertexColor;\n"
	"void main(void) {\n"
	"	vVertexColor = vColor;\n"
	"	gl_Position = mvpMatrix*vVertex;\n"
	"	vec3 vPosition3 = vVertex.xyz / vVertex.w;\n"
	"	vPosition3.z -= -1.5f;\n"
	"	vec3 vLightPosition = vec3(0.0f,0.0f,0.0f);\n"
	"	vec3 aux = vec3(vLightPosition-vPosition3);\n"
	"	lightDir = normalize(aux);\n"
	"	normal = vNormal;\n"
	"}";

static const char *DumbFragmentShader = "#version 140\n"
	"smooth in vec3 normal,lightDir;\n"
	"in vec4 vVertexColor;\n"
	"out vec4 vFragColor;\n"
	"void main(void) {\n"
	"	vec3 n;\n"
	"	float NdotL,NdotHV;\n"
	"	vFragColor = vVertexColor;\n"
	"	vFragColor.rgb *= 0.3f;\n"
	"	n = normalize(normal);\n"
	"	NdotL = max(dot(n,normalize(lightDir)),0.0);\n"
	"	if (NdotL > 0.0) {\n"
	"		vFragColor.rgb += NdotL * vVertexColor.rgb*0.7f;\n"
	"		float fSpec = pow(NdotL, 128.0);\n"
	"		vFragColor.rgb += vec3(fSpec, fSpec, fSpec); }\n"
	"}";

GLuint createProgramWithAttributes(const char *vertexShader, const char *fragmentShader, const GLchar **varying_names, std::vector<std::string> &attributes)
{	// return program # if successful, 0 if fails
	GLuint ret=0,hVertexShader,hFragmentShader;
	GLint testVal;
	hVertexShader = glCreateShader(GL_VERTEX_SHADER);
	GLchar *fsStringPtr[1];
	fsStringPtr[0] = (GLchar *)vertexShader;
	glShaderSource(hVertexShader, 1, (const GLchar **)fsStringPtr, NULL);
	glCompileShader(hVertexShader);
	glGetShaderiv(hVertexShader, GL_COMPILE_STATUS, &testVal);
	if(testVal == GL_FALSE){
		GLchar infoLog[2000];
		GLsizei infoLength;
		glGetShaderInfoLog(hVertexShader,2000,&infoLength,infoLog);
		printf("%s",infoLog);
		glDeleteShader(hVertexShader);
		return ret;
	}
	if(fragmentShader!=NULL)	{
		hFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		fsStringPtr[0] = (GLchar *)fragmentShader;
		glShaderSource(hFragmentShader, 1, (const GLchar **)fsStringPtr, NULL);
		glCompileShader(hFragmentShader);
		glGetShaderiv(hFragmentShader, GL_COMPILE_STATUS, &testVal);
		if(testVal == GL_FALSE){
			GLchar infoLog[2000];
			GLsizei infoLength;
			glGetShaderInfoLog(hFragmentShader,2000,&infoLength,infoLog);
			printf("%s",infoLog);
			glDeleteShader(hVertexShader);
			glDeleteShader(hFragmentShader);
			return ret;
		}
	}
	ret = glCreateProgram();
	glAttachShader(ret, hVertexShader);
	if(fragmentShader!=NULL)	{
		glAttachShader(ret, hFragmentShader);
		for(int i=0; i<(int)attributes.size(); ++i)
			glBindAttribLocation(ret, i, attributes[i].c_str());
	}
	if(varying_names!=NULL)
		glTransformFeedbackVaryings(ret, 2, varying_names, GL_SEPARATE_ATTRIBS);
	glLinkProgram(ret);
	glDeleteShader(hVertexShader);
	if(fragmentShader!=NULL)
		glDeleteShader(hFragmentShader);  
	glGetProgramiv(ret, GL_LINK_STATUS, &testVal);
	if(testVal == GL_FALSE){
		GLchar infoLog[2000];
		GLsizei infoLength;
		glGetProgramInfoLog(ret,2000,&infoLength,infoLog);
		printf("%s",infoLog);
		glDeleteProgram(ret);
		return 0;
	}
	return ret;
}

void setVerticesComputeNormals()
{   // Load the _Positions on the graphics card first, then compute normals and distribute positions
	glBindBuffer(GL_TEXTURE_BUFFER_ARB, _texBOBuffers[0]);
	glBufferData(GL_TEXTURE_BUFFER_ARB, sizeof(GLfloat)*4*_nPositions, &(_Positions[0]), GL_DYNAMIC_DRAW);
	glUseProgram(_NormalMakerProgram);
	glUniform1i( glGetUniformLocation( _NormalMakerProgram, "positionSampler"), 1);
	glUniform1i( glGetUniformLocation( _NormalMakerProgram, "neighborSampler"), 2);
	// bind the texture buffer objects containing _Positions and the topology
	glActiveTexture( GL_TEXTURE1);
	glBindTexture( GL_TEXTURE_BUFFER_EXT, _textureBufferObjects[0]);
	glActiveTexture( GL_TEXTURE2);
	glBindTexture( GL_TEXTURE_BUFFER_EXT, _textureBufferObjects[1]);
	glEnable(GL_RASTERIZER_DISCARD);
	glBeginTransformFeedback(GL_POINTS);
	glDrawArrays(GL_POINTS, 0, _nVertices);
	glEndTransformFeedback();
	glDisable(GL_RASTERIZER_DISCARD);
/*	glBindBuffer(GL_ARRAY_BUFFER, _bufferObjects[1]);
	std::vector<GLfloat> fbCoords;
	fbCoords.assign(_nVertices*3,0.0f);
	glGetBufferSubData(GL_ARRAY_BUFFER,0,sizeof(GLfloat)*_nVertices*3,&(fbCoords[0]));
	glBindBuffer(GL_ARRAY_BUFFER, _bufferObjects[0]);
	fbCoords.assign(_nVertices*4,0.0f);
	glGetBufferSubData(GL_ARRAY_BUFFER,0,sizeof(GLfloat)*_nVertices*4,&(fbCoords[0])); */
}

void initExample()
{
	GLenum err = glewInit();
	if (GLEW_OK != err)	{
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		exit(-1);	}
	fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
	const char *varying_names[]={"vSurfaceNormal","vSurfaceCoord"};
	std::vector<std::string> nada;
	_NormalMakerProgram = createProgramWithAttributes(GTNormalMakerVertexShader,NULL,(const GLchar **)varying_names,nada);
	std::vector<std::string> att;
	att.push_back(std::string("vVertex"));
	att.push_back(std::string("vNormal"));
	att.push_back(std::string("vColor"));
	_RenderingProgram = createProgramWithAttributes(DumbVertexShader,DumbFragmentShader,NULL,att);
	glGenVertexArrays(1,&_vertexArrayBufferObject);
    glGenBuffers(3, _bufferObjects);
	// Create 2 new texture buffer objects
	glGenTextures(2,_textureBufferObjects);
	glGenBuffers(2,_texBOBuffers);
	_Positions.assign(tetVerts,tetVerts+16);
	// topological quad for each vertex with first index its coord in position array
	// and next two position indices of neighbors in counterclockwise order. Last not used.
	GLuint glslNeighbors[] = {0,1,2,0,   1,2,0,0,   2,0,1,0,
		1,3,2,0,    3,2,1,0,    2,1,3,0,
		0,2,3,0,    2,3,0,0,    3,0,2,0,
		0,3,1,0,    3,1,0,0,    1,0,3,0};
	std::vector<GLfloat> colors;
	colors.assign(_nVertices*4,1.0f);
	for(int i=0; i<_nVertices; ++i)	{
		colors[(i<<2)] = (i/4)/2.0f;
		colors[(i<<2)+1] = (2-i/4)/2.0f;
		colors[(i<<2)+2] = (i%4)/3.0f;
	}
    glBindBuffer(GL_ARRAY_BUFFER, _bufferObjects[0]);	// VERTEX_DATA
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*_nVertices*4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, _bufferObjects[1]);	// NORMAL_DATA
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*_nVertices*3, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, _bufferObjects[2]);	// COLOR_DATA
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*_nVertices*4, &(colors[0]), GL_STATIC_DRAW);
	// Create the master vertex array object
	glBindVertexArray(_vertexArrayBufferObject);
    glBindBuffer(GL_ARRAY_BUFFER, _bufferObjects[0]);	// VERTEX_DATA
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, _bufferObjects[1]);	// NORMAL_DATA
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, _bufferObjects[2]);	// COLOR_DATA
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, 0);
    // Unbind to anybody
	glBindVertexArray(0);
	// transform feedback buffers
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER,0,_bufferObjects[1]);
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER,1,_bufferObjects[0]);
	// Set up the two texture buffer objects. Load the _Positions first
	glBindBuffer(GL_TEXTURE_BUFFER_ARB, _texBOBuffers[0]);
	glBufferData(GL_TEXTURE_BUFFER_ARB, sizeof(GLfloat)*4*_nPositions, &(_Positions[0]), GL_DYNAMIC_DRAW);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_BUFFER_ARB, _textureBufferObjects[0]);
	glTexBufferARB(GL_TEXTURE_BUFFER_ARB, GL_RGBA32F, _texBOBuffers[0]); 
	// Load the counterclockwise neighbor info second
	glBindBuffer(GL_TEXTURE_BUFFER_ARB, _texBOBuffers[1]);
	glBufferData(GL_TEXTURE_BUFFER_ARB, sizeof(GLuint)*4*_nVertices, &(glslNeighbors[0]), GL_STATIC_DRAW);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_BUFFER_ARB, _textureBufferObjects[1]);
	glTexBufferARB(GL_TEXTURE_BUFFER_ARB, GL_RGBA32UI, _texBOBuffers[1]);
	setVerticesComputeNormals();
}

void resize(int width, int height)
{
	screenX=width; screenY=height;
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f );
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
    glViewport(0, 0, (GLint) width, (GLint) height);
	float fAspect=(float)width/height;
	for(int i=0; i<16; ++i)
		viewMatrix[i]=0.0f;
	viewMatrix[0] = 3.2327f/fAspect;
	viewMatrix[5] = 3.2327f;
	viewMatrix[10] = -3.1552f;
	viewMatrix[11] = -1.0f;
	viewMatrix[14] = 2.2501f;
	viewMatrix[15] = 7.0991f;
	if(_RenderingProgram)
		glUniformMatrix4fv(glGetUniformLocation(_RenderingProgram, "mvpMatrix"), 1, GL_FALSE, (GLfloat *)viewMatrix);
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if(_vertexArrayBufferObject>0)	{
		glUseProgram(_RenderingProgram);
	    // bind the texture buffer objects containing the normals and the vertices
		glUniformMatrix4fv(glGetUniformLocation(_RenderingProgram, "mvpMatrix"), 1, GL_FALSE, (GLfloat *)viewMatrix);
		glBindVertexArray(_vertexArrayBufferObject);
		glDrawArrays(GL_TRIANGLES, 0, 12);
		glBindVertexArray(0);
	}
    glutSwapBuffers();
}

void changePositions()
{
	if(tetVerts[0]<-2.0f || tetVerts[0]>-0.2f)
		xIncr = -xIncr;
	if(tetVerts[14]<-2.0f || tetVerts[14]>-0.2f)
		zIncr = -zIncr;
	float z,cx,sx,cy,sy;
	cx=(float)cos(xrot); sx=(float)sin(xrot);
	cy=(float)cos(yrot); sy=(float)sin(yrot);
	for(size_t i=0; i<16; i+=4)	{
		if(tetVerts[i]>0.1f)
			tetVerts[i]+=xIncr;
		if(tetVerts[i]<-0.1f)
			tetVerts[i]-=xIncr;
		if(tetVerts[i+2]>0.1f)
			tetVerts[i+2]+=zIncr;
		if(tetVerts[i+2]<-0.1f)
			tetVerts[i+2]-=zIncr;
		z = sx*tetVerts[i+1] + cx*tetVerts[i+2];
		_Positions[i+1] = cx*tetVerts[i+1] - sx*tetVerts[i+2];
		_Positions[i+2] = cy*z - sy*tetVerts[i];
		_Positions[i] = sy*z + cy*tetVerts[i];
	}
	setVerticesComputeNormals();
    glutPostRedisplay();
}

void mouseButton(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON || button == GLUT_RIGHT_BUTTON) {
		if (state == GLUT_DOWN) {
			lastX = x;
			lastY = y;
		}
	}
}

void mouseMove(int x, int y) {
	float fx=(float)(x-lastX)/screenX,fy=(float)(y-lastY)/screenY;
	yrot += 3.1416f*fx;
	xrot += 3.1416f*fy;
	lastX=x;
	lastY=y;
}

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitWindowSize(640,480);
    glutInitWindowPosition(10,10);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutCreateWindow("glsl Normal Computation");
    glutReshapeFunc(resize);
    glutDisplayFunc(display);
    glutIdleFunc(changePositions);
	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMove);
	initExample();
    glutMainLoop();
}
