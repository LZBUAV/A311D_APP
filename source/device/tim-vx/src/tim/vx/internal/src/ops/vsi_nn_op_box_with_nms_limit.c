/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/
#include <string.h>
#include <stdlib.h>

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "kernel/vsi_nn_kernel.h"

#define _INPUT_NUM          (3)
#define _OUTPUT_NUM         (4)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_param_t * param = NULL;
    vsi_nn_box_with_nms_limit_param *p = &self->nn_param.box_with_nms_limit;

    param = vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_float32( param, "score_threshold",  p->score_threshold );
    vsi_nn_kernel_param_add_int32( param, "max_num_detections",  p->max_num_bbox );
    vsi_nn_kernel_param_add_int32( param, "nms_kernel_method",  p->nms_kernel_method );
    vsi_nn_kernel_param_add_float32( param, "iou_threshold",  p->iou_threshold );
    vsi_nn_kernel_param_add_float32( param, "sigma",  p->sigma );
    vsi_nn_kernel_param_add_float32( param, "nms_score_threshold",  p->nms_score_threshold );

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
        "box_with_nms_limit",
        inputs, _INPUT_NUM,
        outputs, _OUTPUT_NUM, param );

    if( self->n )
    {
        status = VSI_SUCCESS;
    }

    vsi_nn_kernel_param_release( &param );

    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /*TODO: Check tensor shapes. */
    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = 1;
        outputs[0]->attr.size[0] = inputs[0]->attr.size[1];
    }

    if( VSI_NN_DIM_AUTO == outputs[1]->attr.dim_num )
    {
        outputs[1]->attr.dim_num = 2;
        outputs[1]->attr.size[0] = 4;
        outputs[1]->attr.size[1] = inputs[0]->attr.size[1];
    }

    if( VSI_NN_DIM_AUTO == outputs[2]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = 1;
        outputs[0]->attr.size[0] = inputs[0]->attr.size[1];
    }

    if( VSI_NN_DIM_AUTO == outputs[3]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = 1;
        outputs[0]->attr.size[0] = inputs[0]->attr.size[1];
    }

    return TRUE;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ BOX_WITH_NMS_LIMIT,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif