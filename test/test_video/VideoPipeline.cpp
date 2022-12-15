/*
 * @Description: Decode stream pipeline.
 * @version: 2.2
 * @Author: Ricardo Lu<shenglu1202@163.com>
 * @Date: 2022-10-11 10:46:02
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-12-15 18:59:34
 */

#include <memory>
#include <functional>

#include <opencv2/opencv.hpp>

#include "Logger.h"
#include "VideoPipeline.h"

static GstFlowReturn cb_appsink_new_sample(
    GstElement* appsink,
    gpointer user_data)
{
    const GstStructure* info = NULL;
    GstBuffer* buffer = NULL;
    GstMapInfo map;
    GstSample* sample = NULL;
    GstCaps* caps = NULL;
    int sample_width = 0;
    int sample_height = 0;

    VideoPipeline* vp = static_cast<VideoPipeline*>(user_data);

    if (!vp->dump) {
        GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(vp->pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "video-pipeline");
        vp->dump = true;
    }

    g_signal_emit_by_name(appsink, "pull-sample", &sample);
    if (!sample) {
        return GST_FLOW_OK;
    } else {
        buffer = gst_sample_get_buffer(sample);
        if (buffer == NULL) {
            LOG_ERROR("Can't get buffer from sample.");
            goto err;
        }
        gst_buffer_map(buffer, &map, GST_MAP_READ);

        caps = gst_sample_get_caps(sample);
        if (caps == NULL) {
            LOG_ERROR("Can't get caps from sample.");
            goto err;
        }

        info = gst_caps_get_structure(caps, 0);
        if (info == NULL) {
            LOG_ERROR("Can't get info from sample.");
            goto err;
        }

        // ---- Read frame and convert to opencv format ---------------
        // convert gstreamer data to OpenCV Mat, you could actually
        // resolve height / width from caps...
        gst_structure_get_int(info, "width", &sample_width);
        gst_structure_get_int(info, "height", &sample_height);

        // appsink algorithm productor queue produce
        {
            // init a tmpMat with gst buffer address: deep copy
            cv::Mat tmpMat(sample_height, sample_width, CV_8UC3, (unsigned char*)map.data, cv::Mat::AUTO_STEP);
            tmpMat = tmpMat.clone();
            vp->productQueue->product(std::make_shared<cv::Mat>(tmpMat));
        }
    }

err:
    if (buffer) {
        gst_buffer_unmap(buffer, &map);
    }
    if (sample) {
        gst_sample_unref(sample);
    }
    return GST_FLOW_OK;
}

VideoPipeline::VideoPipeline(const VideoPipelineConfig& config)
{
    this->config = config;
    dump = false;
}

VideoPipeline::~VideoPipeline(void)
{
    Destroy();
}

static void cb_decodebin_child_added(GstChildProxy* child_proxy, GObject* object,
    gchar* name, gpointer user_data)
{
    VideoPipeline* vp = static_cast<VideoPipeline*>(user_data);

    LOG_INFO("cb_decodebin_child_added called('{}' added)", name);

done:
    return;
}

static void cb_uridecodebin_source_setup(GstElement* object, GstElement* source,
    gpointer user_data)
{
    LOG_INFO("cb_uridecodebin_source_setup called");
}

static void cb_uridecodebin_pad_added(
    GstElement* decodebin,
    GstPad* pad,
    gpointer user_data)
{
    VideoPipeline* vp = static_cast<VideoPipeline*>(user_data);
    GstPad* sinkpad = NULL;

    GstCaps* caps = gst_pad_query_caps(pad, NULL);
    const GstStructure* str = gst_caps_get_structure(caps, 0);
    const gchar* name = gst_structure_get_name(str);
    
    LOG_INFO("cb_uridecodebin_pad_added called {}", name);
    LOG_INFO("structure:{}", gst_structure_to_string(str));

    if (g_str_has_prefix (name, "video/x-raw")) {
        sinkpad = gst_element_get_static_pad(vp->queue, "sink");

        if (sinkpad && gst_pad_link(pad, sinkpad) == GST_PAD_LINK_OK) {
            LOG_INFO("Success to link uridecodebin into pipeline");
        } else {
            LOG_ERROR("Failed to link uridecodebin to pipeline");
        }

        if (sinkpad) {
            gst_object_unref(sinkpad);
        }
    }

    gst_caps_unref(caps);
}

static void cb_uridecodebin_child_added(GstChildProxy* child_proxy,
    GObject* object, gchar* name, gpointer user_data)
{
    VideoPipeline* vp = static_cast<VideoPipeline*>(user_data);

    LOG_INFO("cb_uridecodebin_child_added called('{}' added)", name);

    if (g_strrstr(name, "decodebin") == name) {
        g_signal_connect(G_OBJECT(object), "child-added",
            G_CALLBACK(cb_decodebin_child_added), vp);
    }

done:
    return;
}

bool VideoPipeline::Create(void)
{
    GstCaps* caps;
    GstCapsFeatures* feature;
    gchar* capstr;

    if (!(pipeline = gst_pipeline_new("video-pipeline"))) {
        LOG_ERROR("Failed to create pipeline named video-pipeline");
        goto exit;
    }
    gst_pipeline_set_auto_flush_bus(GST_PIPELINE(pipeline), true);

    if (!(source = gst_element_factory_make("uridecodebin", "src"))) {
        LOG_ERROR("Failed to create element uridecodebin named src");
        goto exit;
    }
    g_object_set (G_OBJECT(source), "uri", config.url.data(), NULL);
    g_signal_connect(G_OBJECT(source), "source-setup", G_CALLBACK(
        cb_uridecodebin_source_setup), this);
    g_signal_connect(G_OBJECT(source), "pad-added",    G_CALLBACK(
        cb_uridecodebin_pad_added),    this);
    g_signal_connect(G_OBJECT(source), "child-added",  G_CALLBACK(
        cb_uridecodebin_child_added),  this);
    gst_bin_add_many(GST_BIN(pipeline), source, NULL);

    if (!(queue = gst_element_factory_make("queue", "queue0"))) {
        LOG_ERROR("Failed to create element queue named queue0");
        goto exit;
    }
    gst_bin_add_many(GST_BIN(pipeline), queue, NULL);

    if (!(converter = gst_element_factory_make("qtivtransform", "videocvt"))) {
        LOG_ERROR("Failed to create element qtivtransform named videocvt");
        goto exit;
    }
    gst_bin_add_many(GST_BIN(pipeline), converter, NULL);

    if (!(convFilter = gst_element_factory_make("capsfilter", "convFilter"))) {
        LOG_ERROR("Failed to create element capsfilter named convFilter");
        goto exit;
    }
    caps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "RGB",
        "width", G_TYPE_INT, config.streamWidth,
        "height", G_TYPE_INT, config.streamHeight, NULL);
    gst_caps_set_features(caps, 0, gst_caps_features_new("memory:GBM", NULL));
    capstr = gst_caps_to_string (caps);
    LOG_INFO("capfilter: {}", capstr);
    g_object_set(G_OBJECT(convFilter), "caps", caps, NULL);
    gst_caps_unref(caps);
    gst_bin_add_many(GST_BIN(pipeline), convFilter, NULL);
    g_free (capstr);

    if (!(appsink = gst_element_factory_make("appsink", "appsink"))) {
        LOG_ERROR("Failed to create element appsink named appsink");
        goto exit;
    }
    g_object_set(appsink, "emit-signals", true, NULL);
    g_object_set(appsink, "drop", config.isDropBuffer, "max-buffers", 1, NULL);
    g_object_set(appsink, "sync", config.isSync, NULL);
    g_signal_connect(appsink, "new-sample", G_CALLBACK(cb_appsink_new_sample), static_cast<void*>(this));
    gst_bin_add_many(GST_BIN(pipeline), appsink, NULL);

    if (!gst_element_link_many(queue, converter, convFilter, appsink, NULL)) {
        LOG_ERROR("Failed to link uridecodebin->queue->qtivtransform->capfilter->appsink");
        goto exit;
    }

    return true;

exit:
    LOG_ERROR("Failed to create video pipeline");
    return false;
}

static GstBusSyncReply
cb_bus_sync_handler(GstBus* bus, GstMessage* message, gpointer data)
{
    if (FALSE) {
        const GstStructure* s = gst_message_get_structure(message);
        guint32 seqnum = gst_message_get_seqnum(message);
        GstObject* src_obj = GST_MESSAGE_SRC(message);
        std::string elem_name = GST_ELEMENT_NAME(src_obj);
        std::string msg_type = GST_MESSAGE_TYPE_NAME(message);

        if (GST_IS_ELEMENT(src_obj) &&(0 == elem_name.compare("appsink")) &&(0 != msg_type.compare("tag"))) {
            LOG_INFO("Got message #{} from element \"{}}\"({}): ",
            (guint) seqnum, GST_ELEMENT_NAME(src_obj),
                GST_MESSAGE_TYPE_NAME(message));
        } else {
            return GST_BUS_PASS;
        }

        if (s) {
            gchar* sstr = gst_structure_to_string(s);
            LOG_INFO("{}", sstr);
            g_free(sstr);
        } else {
            LOG_INFO("no message details");
        }
    }
    
    return GST_BUS_PASS;
}

static gboolean
cb_bus_handler(GstBus* bus, GstMessage* message, gpointer data)
{
    GstElement *pipeline =(GstElement *) data;
    
    switch(GST_MESSAGE_TYPE(message)) {
        case GST_MESSAGE_ERROR:{
            gchar*  debug = NULL;
            GError* err = NULL;
            
            gchar* name = gst_object_get_path_string(message->src);
            gst_message_parse_error(message, &err, &debug);
            LOG_ERROR("ERROR: from element {}", name);
            LOG_ERROR("{}", err->message);
            if (debug != NULL) {
              LOG_ERROR("Debug info:{}", debug);
            }
            
            g_clear_error(&err);
            g_free(debug);
            g_free(name);
            break;
        }
        case GST_MESSAGE_STATE_CHANGED:
        default:
            break;
    }

    return TRUE;
}

bool VideoPipeline::Start(void)
{
    GstBus* bus = gst_element_get_bus(pipeline);
    gst_bus_set_sync_handler(bus, cb_bus_sync_handler, (gpointer)pipeline, NULL);
    bus_watch_id = gst_bus_add_watch(bus, cb_bus_handler, (gpointer)pipeline);
    gst_object_unref(bus);

    if (GST_STATE_CHANGE_FAILURE == gst_element_set_state(pipeline,
        GST_STATE_PLAYING)) {
        LOG_ERROR("Failed to set pipeline to playing state");
        return false;
    }

    return true;
}

void VideoPipeline::Destroy(void)
{
    if (pipeline) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
        pipeline = NULL;
    }
}

void VideoPipeline::SetUserData(std::shared_ptr<SafetyQueue<cv::Mat>> user_data)
{
    productQueue = user_data;
}