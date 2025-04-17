function[mic] = define_mk32()
%DEFINE_Mk32 creates a structure representing MK32 microphone
%
%[mic] = define_mk32()
%
%mic has the following fields:
%              a: radius of sphere in metres
%  sensor_angles: [32 x 2] matrix specifying angle of each transducer in
%                 Daniel's [azimuth, inclination] coordinates system
%        sphType: 'rigid' or 'open'
%         config: 'mk32' (fixed)
%
%requires: sphDist from SPHERICAL_TOOLBOX
%

mic.ShOrder = 4;
mic.radius = 0.042;
mic.config = 'mk32';
sensor_positions = [0,-27.4426,	-31.7947;
7.1055,	-40.0436,	-10.4892;
29.4026,	-26.4295,	-14.1764;
17.9583,	-11.5855,	-36.1563;
36.996,	-3.2836	-19.6091;
20.8661,	12.7201,	-34.1585;
33.9048,	24.417,	-4.2751;
17.9179,	31.4604,	-21.2884;
-10.3069,	35.3805,	-20.1492;
-4.8306,	18.4079,	-37.4408;
-30.6117,	28.216,	-5.5478;
-31.9675,	10.7677,	-25.0227;
-39.9606,	-8.1693,	-10.0205;
-16.6071,	-3.0242,	-38.4585;
-24.2512,	-24.4464,	-24.0469;
-18.3083,	-37.6586,	-3.2614;
-26.306,	15.214,	28.9919;
-15.7925,	35.8212,	15.2132;
1.3838,	27.0496,	32.0999;
3.5556,	41.8492,	0.0069;
21.5238,	31.9112,	16.8049;
5.5474,	4.9085,	41.3417;
30.4241,	9.4352,	27.3743;
41.5466,	1.8144,	5.8812;
33.4141,	-23.1077,	10.6552;
22.5122,	-13.6203,	32.7367;
-3.6574,	-17.6084,	37.9548;
12.1849,	-37.5092,	14.4425;
-10.3295,	-33.6159,	22.9624;
-33.6155,	-22.9039,	10.4598;
-27.611,	-9.1926	,30.2841;
-39.8367,	9.629,	9.1826];

rot = [  1.0000000,  0.0000000,  0.0000000;
   0.0000000,  0.0000000, 1.0000000;
   0.0000000,  -1.0000000,  0.0000000 ];

sensor_positions = sensor_positions/1000.0;
rot_sensor_positions = rot*sensor_positions';
mic.sensor_positions  = rot_sensor_positions';
%%
[az,inc,~] = mycart2sph(mic.sensor_positions(:,1),mic.sensor_positions(:,2),mic.sensor_positions(:,3));
%azimuth from -pi to pi (instead of 0 to 2*pi) where left is pi/2 and right is -pi/2
mic.sensor_angle(:,1) =  az;
mic.sensor_angle(:,2) = pi/2 - inc;
mic.sphType = 'rigid';