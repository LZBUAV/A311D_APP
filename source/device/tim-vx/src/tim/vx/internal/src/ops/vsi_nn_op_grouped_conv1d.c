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
#include "vsi_nn_log.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "vsi_nn_internal_node.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_constraint_check.h"

#define _INPUT_NUM          (3)
#define _OUTPUT_NUM         (1)

static vsi_nn_tensor_t * _expand_tensor_dim
    ( vsi_nn_graph_t * graph, vsi_nn_tensor_t *tensor, uint32_t * shape, size_t rank, int32_t expand_dim )
{
    uint32_t new_shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    uint32_t i, cnt;
    if ( expand_dim < 0 )
    {
        expand_dim = (int32_t)rank + expand_dim;
    }
    if ( expand_dim < 0 || (uint32_t)expand_dim > rank )
    {
        VSILOGE("Run dim to expand %d, rank is %lu", expand_dim, rank);
        return NULL;
    }
    for ( i = 0, cnt = 0; i < rank; i ++ )
    {
        if ( i == (uint32_t)expand_dim )
        {
            new_shape[cnt] = 1;
            cnt ++;
        }
        new_shape[cnt] = shape[i];
        cnt ++;
    }
    if ( (uint32_t)expand_dim == rank )
    {
        new_shape[cnt] = 1;
    }

    return vsi_nn_reshape_tensor( graph, tensor, new_shape, (uint32_t)rank + 1 );
} /* _expand_tensor_dim() */

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return vsi_nn_internal_compute_node( self );
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;

    ret = vsi_nn_OpCheck(VSI_NN_OP_CONV2D, self, inputs, outputs);

    return ret;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_grouped_conv1d_param* p = &self->nn_param.grouped_conv1d;

    vsi_nn_internal_init_node_wksp(self);

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.size[0] = vsi_nn_ComputeFilterSize
            (
            inputs[0]->attr.size[0],
            inputs[1]->attr.size[0],
            p->pad,
            p->stride,
            p->dilation,
            VSI_NN_ROUND_FLOOR
            );

        outputs[0]->attr.size[1] = inputs[1]->attr.size[2];
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
        outputs[0]->attr.dim_num = 3;
    }

    p->local->input = _expand_tensor_dim( self->graph, inputs[0],
            inputs[0]->attr.size, inputs[0]->attr.dim_num, 0 );
    p->local->weight = _expand_tensor_dim( self->graph, inputs[1],
            inputs[1]->attr.size, inputs[1]->attr.dim_num, 0 );
    p->local->output = _expand_tensor_dim( self->graph, outputs[0],
            outputs[0]->attr.size, outputs[0]->attr.dim_num, 0 );


    curr = vsi_nn_internal_new_node(self, VSI_NN_OP_GROUPED_CONV2D, 0, 0);
    curr->inputs[0] = p->local->input;
    curr->inputs[1] = p->local->weight;
    curr->inputs[2] = inputs[2];
    curr->outputs[0] = p->local->output;
    curr->node->nn_param.grouped_conv2d.ksize[0] = 1;
    curr->node->nn_param.grouped_conv2d.ksize[1] = p->ksize;
    curr->node->nn_param.grouped_conv2d.dilation[0] = 1;
    curr->node->nn_param.grouped_conv2d.dilation[1] = p->dilation;
    curr->node->nn_param.grouped_conv2d.pad[0] = 0;
    curr->node->nn_param.grouped_conv2d.pad[1] = p->pad[0];
    curr->node->nn_param.grouped_conv2d.pad[2] = 0;
    curr->node->nn_param.grouped_conv2d.pad[3] = p->pad[1];
    curr->node->nn_param.grouped_conv2d.stride[0] = 1;
    curr->node->nn_param.grouped_conv2d.stride[1] = p->stride;
    curr->node->nn_param.grouped_conv2d.group = p->group;
    curr->node->nn_param.grouped_conv2d.multiplier = p->multiplier;
    curr->node->nn_param.grouped_conv2d.weights = p->weights;
    curr->node->nn_param.grouped_conv2d.pad_type = p->pad_type;

    vsi_nn_internal_setup_node(self, curr);

    return TRUE;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t* self
    )
{

    self->nn_param.grouped_conv1d.local = (grouped_conv1d_local_data_t *)malloc(sizeof(grouped_conv1d_local_data_t));
    memset(self->nn_param.grouped_conv1d.local, 0x00, sizeof(grouped_conv1d_local_data_t));

    return VSI_SUCCESS;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t* self
    )
{
    vsi_nn_grouped_conv1d_param* p = &self->nn_param.grouped_conv1d;
    vsi_nn_internal_deinit_node_wksp(self);

    vsi_safe_release_tensor(p->local->input);
    vsi_safe_release_tensor(p->local->weight);
    vsi_safe_release_tensor(p->local->output);
    vsi_nn_safe_free(p->local);

    return vsi_nn_op_common_deinit(self);
} /* op_deinit() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ GROUPED_CONV1D,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS

