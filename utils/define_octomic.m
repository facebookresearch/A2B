function[mic] = define_octomic()
%DEFINE_OCTOMIC creates a structure representing CoreSound Octomic
%
%[mic] = define_octomic()
%
%mic has the following fields:
%              a: radius of sphere in metres
%  sensor_angles: [8 x 2] matrix specifying angle of each transducer in
%                 Daniel's [azimuth, inclination] coordinates system
%        sphType: 'rigid' or 'open'
%         config: 'octomic' (fixed)
%
%requires: sphDist from SPHERICAL_TOOLBOX
%

mic.ShOrder = 2;
mic.radius = 0.006;
mic.config = 'octomic';
d = mic.radius*sqrt(3.0)*0.5;
%Top ring of capsules: #1, #3, #5 and #7
%Bottom ring of capsules: #2, #4, #6 and #8
sensor_positions = [
    [-d, -d,  -d];
    [-d, -d,  d];
    [-d,  d,  -d];
    [-d,  d,  d];
    [ d, -d, -d];
    [ d, -d,  d];
    [ d,  d, -d];
    [ d,  d,  d]];

mic.sensor_positions  = sensor_positions;
%%
[az,inc,~] = mycart2sph(mic.sensor_positions(:,1),mic.sensor_positions(:,2),mic.sensor_positions(:,3));
%azimuth from -pi to pi (instead of 0 to 2*pi) where left is pi/2 and right is -pi/2
mic.sensor_angle(:,1) =  az;
mic.sensor_angle(:,2) = pi/2 - inc;
mic.sphType = 'rigid';