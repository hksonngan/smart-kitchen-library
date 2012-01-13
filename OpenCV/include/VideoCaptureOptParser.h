#ifdef USE_VIDEO_CAPTURE_OPT_PARSER
#ifndef __SKL__VIDEO_CAPTURE_OPT_PARSER_H__
#define __SKL__VIDEO_CAPTURE_OPT_PARSER_H__

#define opt_on_cam_prop(PROP_NAME) opt_on(double, PROP_NAME, 0.0,"","<PROP_VAL>", "set camera parameter PROP_NAME")

opt_on_cam_prop(POS_MSEC);
opt_on_cam_prop(POS_FRAMES);
opt_on_cam_prop(POS_AVI_RATIO);
opt_on_cam_prop(FRAME_WIDTH);
opt_on_cam_prop(FRAME_HEIGHT);
opt_on_cam_prop(FPS);
opt_on_cam_prop(FOURCC);
opt_on_cam_prop(FRAME_COUNT);
opt_on_cam_prop(FORMAT);
opt_on_cam_prop(MODE);
opt_on_cam_prop(BRIGHTNESS);
opt_on_cam_prop(CONTRAST);
opt_on_cam_prop(SATURATION);
opt_on_cam_prop(HUE);
opt_on_cam_prop(GAIN);
opt_on_cam_prop(EXPOSURE);
opt_on_cam_prop(CONVERT_RGB);
//	opt_on_cam_prop(WHITE_BALANCE); // reserved parameter by OpenCV
opt_on_cam_prop(RECTIFICATION);

#define opt_set_cap_prop(prop_name,params) \
if(prop_name!=0.0){ assert(params.set(CV_CAP_PROP_##prop_name,prop_name));}

#define opt_parse_cap_prop(params) \
opt_set_cap_prop(POS_MSEC,params)\
opt_set_cap_prop(POS_FRAMES,params)\
opt_set_cap_prop(POS_AVI_RATIO,params)\
opt_set_cap_prop(FRAME_WIDTH,params)\
opt_set_cap_prop(FRAME_HEIGHT,params)\
opt_set_cap_prop(FPS,params)\
opt_set_cap_prop(FOURCC,params)\
opt_set_cap_prop(FRAME_COUNT,params)\
opt_set_cap_prop(FORMAT,params)\
opt_set_cap_prop(MODE,params)\
opt_set_cap_prop(BRIGHTNESS,params)\
opt_set_cap_prop(CONTRAST,params)\
opt_set_cap_prop(SATURATION,params)\
opt_set_cap_prop(HUE,params)\
opt_set_cap_prop(GAIN,params)\
opt_set_cap_prop(EXPOSURE,params)\
opt_set_cap_prop(CONVERT_RGB,params)\
opt_set_cap_prop(RECTIFICATION,params)
//	opt_set_cap_prop(WHITE_BALANCE,params)\ // reserved parameter by OpenCV


#endif
#endif
