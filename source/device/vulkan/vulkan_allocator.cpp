/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Parts of the following code in this file refs to
 * https://github.com/Tencent/ncnn/tree/master/src/layer/vulkan/
 * Tencent is pleased to support the open source community by making ncnn
 * available.
 *
 * Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 */

/*
 * Copyright (c) 2020, Open AI Lab
 * Author: ddzhao@openailab.com
 */

#include <stdio.h>
#include "vulkan_allocator.hpp"
#include "vulkan_gpu.hpp"
#include "vulkan_pipeline.hpp"

#include <algorithm>

namespace TEngine {

Allocator::~Allocator()
{
}

VkAllocator::VkAllocator(const GPUDevice* _vkdev)
    : vkdev(_vkdev)
{
    buffer_memory_type_index = (uint32_t)-1;
    image_memory_type_index = (uint32_t)-1;
    mappable = false;
    coherent = false;
}

static inline size_t round_up(size_t n, size_t multiple)
{
    return (n + multiple - 1) / multiple * multiple;
}

static inline size_t round_down(size_t n, size_t multiple)
{
    return n / multiple * multiple;
}

static inline size_t least_common_multiple(size_t a, size_t b)
{
    if (a == b)
        return a;

    if (a > b)
        return least_common_multiple(b, a);

    size_t lcm = b;
    while (lcm % a != 0)
    {
        lcm += b;
    }

    return lcm;
}

int VkAllocator::flush(VkBufferMemory* ptr)
{
    if (coherent)
        return 0;

    VkMappedMemoryRange mappedMemoryRange;
    mappedMemoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mappedMemoryRange.pNext = 0;
    mappedMemoryRange.memory = ptr->memory;
    mappedMemoryRange.offset = round_down(ptr->offset, vkdev->info.non_coherent_atom_size);
    mappedMemoryRange.size = round_up(ptr->offset + ptr->capacity, vkdev->info.non_coherent_atom_size) - mappedMemoryRange.offset;

    VkResult ret = vkFlushMappedMemoryRanges(vkdev->vkdevice(), 1, &mappedMemoryRange);
    if (ret != VK_SUCCESS)
    {
        printf("vkFlushMappedMemoryRanges failed %d", ret);
        return -1;
    }

    return 0;
}

int VkAllocator::invalidate(VkBufferMemory* ptr)
{
    if (coherent)
        return 0;

    VkMappedMemoryRange mappedMemoryRange;
    mappedMemoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mappedMemoryRange.pNext = 0;
    mappedMemoryRange.memory = ptr->memory;
    mappedMemoryRange.offset = round_down(ptr->offset, vkdev->info.non_coherent_atom_size);
    mappedMemoryRange.size = round_up(ptr->offset + ptr->capacity, vkdev->info.non_coherent_atom_size) - mappedMemoryRange.offset;

    VkResult ret = vkInvalidateMappedMemoryRanges(vkdev->vkdevice(), 1, &mappedMemoryRange);
    if (ret != VK_SUCCESS)
    {
        printf("vkInvalidateMappedMemoryRanges failed %d", ret);
        return -1;
    }
    return 0;
}

VkBuffer VkAllocator::create_buffer(size_t size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo bufferCreateInfo;
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.pNext = 0;
    bufferCreateInfo.flags = 0;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = usage;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferCreateInfo.queueFamilyIndexCount = 0;
    bufferCreateInfo.pQueueFamilyIndices = 0;

    VkBuffer buffer = 0;
    VkResult ret = vkCreateBuffer(vkdev->vkdevice(), &bufferCreateInfo, 0, &buffer);
    if (ret != VK_SUCCESS)
    {
        printf("vkCreateBuffer failed %d", ret);
        return 0;
    }

    return buffer;
}

VkDeviceMemory VkAllocator::allocate_memory(size_t size, uint32_t memory_type_index)
{
    VkMemoryAllocateInfo memoryAllocateInfo;
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.pNext = 0;
    memoryAllocateInfo.allocationSize = size;
    memoryAllocateInfo.memoryTypeIndex = memory_type_index;

    VkDeviceMemory memory = 0;
    VkResult ret = vkAllocateMemory(vkdev->vkdevice(), &memoryAllocateInfo, 0, &memory);
    if (ret != VK_SUCCESS)
    {
        printf("vkAllocateMemory failed %d", ret);
        return 0;
    }
    return memory;
}

VkDeviceMemory VkAllocator::allocate_dedicated_memory(size_t size, uint32_t memory_type_index, VkImage image, VkBuffer buffer)
{
    VkMemoryAllocateInfo memoryAllocateInfo;
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.pNext = 0;
    memoryAllocateInfo.allocationSize = size;
    memoryAllocateInfo.memoryTypeIndex = memory_type_index;

    VkMemoryDedicatedAllocateInfoKHR memoryDedicatedAllocateInfo;
    memoryDedicatedAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR;
    memoryDedicatedAllocateInfo.pNext = 0;
    memoryDedicatedAllocateInfo.image = image;
    memoryDedicatedAllocateInfo.buffer = buffer;
    memoryAllocateInfo.pNext = &memoryDedicatedAllocateInfo;

    VkDeviceMemory memory = 0;
    VkResult ret = vkAllocateMemory(vkdev->vkdevice(), &memoryAllocateInfo, 0, &memory);
    if (ret != VK_SUCCESS)
    {
        printf("vkAllocateMemory failed %d", ret);
        return 0;
    }

    return memory;
}

VkImage VkAllocator::create_image(VkImageType type, int width, int height, int depth, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage)
{
    VkImageCreateInfo imageCreateInfo;
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    imageCreateInfo.pNext = 0;
    imageCreateInfo.flags = 0;
    imageCreateInfo.imageType = type;
    imageCreateInfo.format = format;
    imageCreateInfo.extent.width = width;
    imageCreateInfo.extent.height = height;
    imageCreateInfo.extent.depth = depth;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = tiling;
    imageCreateInfo.usage = usage;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.queueFamilyIndexCount = 0;
    imageCreateInfo.pQueueFamilyIndices = 0;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image;
    VkResult ret = vkCreateImage(vkdev->vkdevice(), &imageCreateInfo, 0, &image);
    if (ret != VK_SUCCESS)
    {
        printf("vkCreateImage failed %d %d %d %d %d %d %d %d", ret, type, width, height, depth, format, tiling, usage);
        return 0;
    }

    return image;
}

VkImageView VkAllocator::create_imageview(VkImageViewType type, VkImage image, VkFormat format)
{
    VkImageViewCreateInfo imageViewCreateInfo;
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.pNext = 0;
    imageViewCreateInfo.flags = 0;
    imageViewCreateInfo.image = image;
    imageViewCreateInfo.viewType = type;
    imageViewCreateInfo.format = format;
    imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount = 1;

    VkImageView imageview;
    VkResult ret = vkCreateImageView(vkdev->vkdevice(), &imageViewCreateInfo, 0, &imageview);
    if (ret != VK_SUCCESS)
    {
        printf("vkCreateImageView failed %d", ret);
        return 0;
    }

    return imageview;
}

VkBlobAllocator::VkBlobAllocator(const GPUDevice* _vkdev)
    : VkAllocator(_vkdev)
{
    buffer_offset_alignment = vkdev->info.buffer_offset_alignment;
    bind_memory_offset_alignment = vkdev->info.buffer_image_granularity;

    if (vkdev->info.type == 1)
    {
        // on integrated gpu, there may be device local only memory too, eg. AMD APU
        // assuming larger alignment always keeps us safe :)

        // least common multiple for memory_map_alignment and buffer_offset_alignment and non_coherent_atom_size
        buffer_offset_alignment = least_common_multiple(buffer_offset_alignment, vkdev->info.memory_map_alignment);
        buffer_offset_alignment = least_common_multiple(buffer_offset_alignment, vkdev->info.non_coherent_atom_size);
    }

    block_size = alignSize(16 * 1024 * 1024, buffer_offset_alignment); // 16M
}

VkBlobAllocator::~VkBlobAllocator()
{
    clear();
}

// TODO
void VkBlobAllocator::clear()
{
    //     TLOG_INFO("VkBlobAllocator %lu", buffer_blocks.size());

    for (size_t i = 0; i < buffer_blocks.size(); i++)
    {
        VkBufferMemory* ptr = buffer_blocks[i];

        //         std::list< std::pair<size_t, size_t> >::iterator it = buffer_budgets[i].begin();
        //         while (it != buffer_budgets[i].end())
        //         {
        //             TLOG_INFO("VkBlobAllocator budget %p %lu %lu", ptr->buffer, it->first, it->second);
        //             it++;
        //         }

        if (mappable)
            vkUnmapMemory(vkdev->vkdevice(), ptr->memory);

        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

        delete ptr;
    }
    buffer_blocks.clear();

    buffer_budgets.clear();

    for (size_t i = 0; i < image_memory_blocks.size(); i++)
    {
        VkDeviceMemory memory = image_memory_blocks[i];

        //         std::list< std::pair<size_t, size_t> >::iterator it = image_memory_budgets[i].begin();
        //         while (it != image_memory_budgets[i].end())
        //         {
        //             TLOG_INFO("VkBlobAllocator budget %p %lu %lu", memory, it->first, it->second);
        //             it++;
        //         }

        vkFreeMemory(vkdev->vkdevice(), memory, 0);
    }
    image_memory_blocks.clear();

    image_memory_budgets.clear();
}

VkBufferMemory* VkBlobAllocator::fastMalloc(size_t size)
{
    size_t aligned_size = alignSize(size, buffer_offset_alignment);

    const int buffer_block_count = buffer_blocks.size();

    // find first spare space in buffer_blocks
    for (int i = 0; i < buffer_block_count; i++)
    {
        std::list<std::pair<size_t, size_t> >::iterator it = buffer_budgets[i].begin();
        while (it != buffer_budgets[i].end())
        {
            size_t budget_size = it->second;
            if (budget_size < aligned_size)
            {
                it++;
                continue;
            }

            // return sub buffer
            VkBufferMemory* ptr = new VkBufferMemory;

            ptr->buffer = buffer_blocks[i]->buffer;
            ptr->offset = it->first;
            ptr->memory = buffer_blocks[i]->memory;
            ptr->capacity = aligned_size;
            ptr->mapped_ptr = buffer_blocks[i]->mapped_ptr;
            ptr->access_flags = 0;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

            // adjust buffer_budgets
            if (budget_size == aligned_size)
            {
                buffer_budgets[i].erase(it);
            }
            else
            {
                it->first += aligned_size;
                it->second -= aligned_size;
            }

            // printf("VkBlobAllocator M %p +%lu %lu\n", ptr->buffer, ptr->offset, ptr->capacity);

            return ptr;
        }
    }

    size_t new_block_size = std::max(block_size, aligned_size);

    // create new block
    VkBufferMemory* block = new VkBufferMemory;

    block->buffer = create_buffer(new_block_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    block->offset = 0;

    // TODO respect VK_KHR_dedicated_allocation ?

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), block->buffer, &memoryRequirements);

    // setup memory type and alignment
    if (buffer_memory_type_index == (uint32_t)-1)
    {
        if (vkdev->info.type == 1)
        {
            // integrated gpu, prefer unified memory
            buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
        }
        else
        {
            // discrete gpu, device local
            buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        }

        mappable = vkdev->is_mappable(buffer_memory_type_index);
        coherent = vkdev->is_coherent(buffer_memory_type_index);
    }

    block->memory = allocate_memory(memoryRequirements.size, buffer_memory_type_index);

    // ignore memoryRequirements.alignment as we always bind at zero offset
    vkBindBufferMemory(vkdev->vkdevice(), block->buffer, block->memory, 0);

    block->mapped_ptr = 0;
    if (mappable)
    {
        vkMapMemory(vkdev->vkdevice(), block->memory, 0, new_block_size, 0, &block->mapped_ptr);
    }

    buffer_blocks.push_back(block);

    // return sub buffer
    VkBufferMemory* ptr = new VkBufferMemory;

    ptr->buffer = block->buffer;
    ptr->offset = 0;
    ptr->memory = block->memory;
    ptr->capacity = aligned_size;
    ptr->mapped_ptr = block->mapped_ptr;
    ptr->access_flags = 0;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    // adjust buffer_budgets
    std::list<std::pair<size_t, size_t> > budget;
    if (new_block_size > aligned_size)
    {
        budget.push_back(std::make_pair(aligned_size, new_block_size - aligned_size));
    }
    buffer_budgets.push_back(budget);

    //     TLOG_INFO("VkBlobAllocator M %p +%lu %lu", ptr->buffer, ptr->offset, ptr->capacity);

    return ptr;
}

VkImageMemory* VkBlobAllocator::fastMalloc(int dims, int w, int h, int c, size_t elemsize, int elempack)
{
    if (elempack != 1 && elempack != 4 && elempack != 8)
    {
        printf("elempack must be 1 4 8");
        return 0;
    }

    // resolve format
    VkFormat format = VK_FORMAT_UNDEFINED;

    if (elemsize / elempack == 4)
    {
        // fp32
        if (elempack == 1) format = VK_FORMAT_R32_SFLOAT;
        if (elempack == 4) format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (elempack == 8) format = VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    if (elemsize / elempack == 2)
    {
        // fp16
        if (elempack == 1) format = VK_FORMAT_R16_SFLOAT;
        if (elempack == 4) format = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (elempack == 8) format = VK_FORMAT_R16G16B16A16_SFLOAT;
    }

    // resolve image width height depth
    int width = w;
    int height = h;
    int depth = c;

    // large elempack spills on image w
    if (elempack == 8) width *= 2;

    VkImageType image_type;
    VkImageViewType imageview_type;
    if (dims == 1)
    {
        image_type = VK_IMAGE_TYPE_1D;
        imageview_type = VK_IMAGE_VIEW_TYPE_1D;

        if (width > (int)vkdev->info.max_image_dimension_1d)
        {
            printf("image dimension too large %d > %d", width, (int)vkdev->info.max_image_dimension_1d);
            return 0;
        }
    }
    else if (dims == 2)
    {
        image_type = VK_IMAGE_TYPE_2D;
        imageview_type = VK_IMAGE_VIEW_TYPE_2D;

        if (width > (int)vkdev->info.max_image_dimension_2d || height > (int)vkdev->info.max_image_dimension_2d)
        {
            printf("image dimension too large %d %d > %d", width, height, (int)vkdev->info.max_image_dimension_2d);
            return 0;
        }
    }
    else // if (dims == 3)
    {
        image_type = VK_IMAGE_TYPE_3D;
        imageview_type = VK_IMAGE_VIEW_TYPE_3D;

        if (width > (int)vkdev->info.max_image_dimension_3d || height > (int)vkdev->info.max_image_dimension_3d || depth > (int)vkdev->info.max_image_dimension_3d)
        {
            printf("image dimension too large %d %d %d > %d", width, height, depth, (int)vkdev->info.max_image_dimension_3d);
            return 0;
        }
    }

    VkImageMemory* ptr = new VkImageMemory;

    ptr->image = create_image(image_type, width, height, depth, format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    ptr->image_type = image_type;
    ptr->imageview_type = imageview_type;
    ptr->width = width;
    ptr->height = height;
    ptr->depth = depth;
    ptr->format = format;

    // TODO respect VK_KHR_dedicated_allocation ?
    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(vkdev->vkdevice(), ptr->image, &memoryRequirements);

    const size_t size = memoryRequirements.size;
    const size_t alignment = std::max((size_t)memoryRequirements.alignment, bind_memory_offset_alignment);

    size_t aligned_size = alignSize(size, alignment);

    const int image_memory_block_count = image_memory_blocks.size();

    // find first spare space in image_memory_blocks
    for (int i = 0; i < image_memory_block_count; i++)
    {
        std::list<std::pair<size_t, size_t> >::iterator it = image_memory_budgets[i].begin();
        while (it != image_memory_budgets[i].end())
        {
            // we cannot use it->first directly for base offset alignment
            size_t bind_base_offset = it->first;
            size_t bind_offset = alignSize(bind_base_offset, alignment);
            size_t budget_size = it->second;
            if (budget_size < aligned_size + (bind_offset - bind_base_offset))
            {
                it++;
                continue;
            }
            // bind at memory offset
            ptr->memory = image_memory_blocks[i];
            ptr->bind_offset = bind_offset;
            ptr->bind_capacity = aligned_size;

            vkBindImageMemory(vkdev->vkdevice(), ptr->image, ptr->memory, ptr->bind_offset);

            // do not allow host access to optimal tiling image
            ptr->mapped_ptr = 0;

            ptr->imageview = create_imageview(imageview_type, ptr->image, format);

            ptr->access_flags = 0;
            ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            ptr->command_refcount = 0;

            if (bind_base_offset != bind_offset)
            {
                // NOTE there is small offset inside bind_base_offset and bind_offset
                // adjust ptr->bind_offset and ptr->bind_capacity after vkBindImageMemory
                // so that memory management could be easier
                aligned_size += (bind_offset - bind_base_offset);

                ptr->bind_offset = bind_base_offset;
                ptr->bind_capacity = aligned_size;
            }

            // adjust image_memory_budgets
            if (budget_size == aligned_size)
            {
                image_memory_budgets[i].erase(it);
            }
            else
            {
                it->first += aligned_size;
                it->second -= aligned_size;
            }

            //             TLOG_INFO("VkBlobAllocator M %p +%lu %lu", ptr->memory, ptr->bind_offset, ptr->bind_capacity);

            return ptr;
        }
    }

    // setup memory type and alignment
    if (image_memory_type_index == (uint32_t)-1)
    {
        if (vkdev->info.type == 1)
        {
            // integrated gpu, prefer unified memory
            image_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
        }
        else
        {
            // discrete gpu, device local
            image_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        }
        mappable = vkdev->is_mappable(image_memory_type_index);
        coherent = vkdev->is_coherent(image_memory_type_index);
    }

    // create new block
    size_t new_block_size = std::max(block_size, aligned_size);

    // bind at memory offset
    ptr->memory = allocate_memory(new_block_size, image_memory_type_index);
    ptr->bind_offset = 0;
    ptr->bind_capacity = aligned_size;

    // ignore memoryRequirements2.memoryRequirements.alignment as we always bind at zero offset
    vkBindImageMemory(vkdev->vkdevice(), ptr->image, ptr->memory, ptr->bind_offset);

    // do not allow host access to optimal tiling image
    ptr->mapped_ptr = 0;

    ptr->imageview = create_imageview(imageview_type, ptr->image, format);

    ptr->access_flags = 0;
    ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    ptr->command_refcount = 0;

    // adjust image_memory_budgets
    image_memory_blocks.push_back(ptr->memory);

    std::list<std::pair<size_t, size_t> > budget;
    if (new_block_size > aligned_size)
    {
        budget.push_back(std::make_pair(aligned_size, new_block_size - aligned_size));
    }
    image_memory_budgets.push_back(budget);

    //     TLOG_INFO("VkBlobAllocator M %p +%lu %lu", ptr->memory, ptr->bind_offset, ptr->bind_capacity);

    return ptr;
}

void VkBlobAllocator::fastFree(VkBufferMemory* ptr)
{
    //     TLOG_INFO("VkBlobAllocator F %p +%lu %lu", ptr->buffer, ptr->offset, ptr->capacity);

    const int buffer_block_count = buffer_blocks.size();

    int block_index = -1;
    for (int i = 0; i < buffer_block_count; i++)
    {
        if (buffer_blocks[i]->buffer == ptr->buffer && buffer_blocks[i]->memory == ptr->memory)
        {
            block_index = i;
            break;
        }
    }

    if (block_index == -1)
    {
        printf("FATAL ERROR! unlocked VkBlobAllocator get wild %p", ptr->buffer);

        delete ptr;

        return;
    }

    // merge
    std::list<std::pair<size_t, size_t> >::iterator it_merge_left = buffer_budgets[block_index].end();
    std::list<std::pair<size_t, size_t> >::iterator it_merge_right = buffer_budgets[block_index].end();
    std::list<std::pair<size_t, size_t> >::iterator it = buffer_budgets[block_index].begin();
    for (; it != buffer_budgets[block_index].end(); it++)
    {
        if (it->first + it->second == ptr->offset)
        {
            it_merge_left = it;
        }
        else if (ptr->offset + ptr->capacity == it->first)
        {
            it_merge_right = it;
        }
    }

    if (it_merge_left != buffer_budgets[block_index].end() && it_merge_right != buffer_budgets[block_index].end())
    {
        it_merge_left->second = it_merge_right->first + it_merge_right->second - it_merge_left->first;
        buffer_budgets[block_index].erase(it_merge_right);
    }
    else if (it_merge_left != buffer_budgets[block_index].end())
    {
        it_merge_left->second = ptr->offset + ptr->capacity - it_merge_left->first;
    }
    else if (it_merge_right != buffer_budgets[block_index].end())
    {
        it_merge_right->second = it_merge_right->first + it_merge_right->second - ptr->offset;
        it_merge_right->first = ptr->offset;
    }
    else
    {
        if (ptr->offset == 0)
        {
            // chain leading block
            buffer_budgets[block_index].push_front(std::make_pair(ptr->offset, ptr->capacity));
        }
        else
        {
            buffer_budgets[block_index].push_back(std::make_pair(ptr->offset, ptr->capacity));
        }
    }

    delete ptr;
}

void VkBlobAllocator::fastFree(VkImageMemory* ptr)
{
    //     TLOG_INFO("VkBlobAllocator F %p +%lu %lu", ptr->memory, ptr->bind_offset, ptr->bind_capacity);

    const int image_memory_block_count = image_memory_blocks.size();

    int block_index = -1;
    for (int i = 0; i < image_memory_block_count; i++)
    {
        if (image_memory_blocks[i] == ptr->memory)
        {
            block_index = i;
            break;
        }
    }

    if (block_index == -1)
    {
        printf("FATAL ERROR! unlocked VkBlobAllocator get wild %p\n", ptr->memory);

        if (!ptr->command_refcount)
        {
            vkDestroyImageView(vkdev->vkdevice(), ptr->imageview, 0);
            vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);

            delete ptr;
        }

        return;
    }

    // merge
    std::list<std::pair<size_t, size_t> >::iterator it_merge_left = image_memory_budgets[block_index].end();
    std::list<std::pair<size_t, size_t> >::iterator it_merge_right = image_memory_budgets[block_index].end();
    std::list<std::pair<size_t, size_t> >::iterator it = image_memory_budgets[block_index].begin();
    for (; it != image_memory_budgets[block_index].end(); it++)
    {
        if (it->first + it->second == ptr->bind_offset)
        {
            it_merge_left = it;
        }
        else if (ptr->bind_offset + ptr->bind_capacity == it->first)
        {
            it_merge_right = it;
        }
    }

    if (it_merge_left != image_memory_budgets[block_index].end() && it_merge_right != image_memory_budgets[block_index].end())
    {
        it_merge_left->second = it_merge_right->first + it_merge_right->second - it_merge_left->first;
        image_memory_budgets[block_index].erase(it_merge_right);
    }
    else if (it_merge_left != image_memory_budgets[block_index].end())
    {
        it_merge_left->second = ptr->bind_offset + ptr->bind_capacity - it_merge_left->first;
    }
    else if (it_merge_right != image_memory_budgets[block_index].end())
    {
        it_merge_right->second = it_merge_right->first + it_merge_right->second - ptr->bind_offset;
        it_merge_right->first = ptr->bind_offset;
    }
    else
    {
        if (ptr->bind_offset == 0)
        {
            // chain leading block
            image_memory_budgets[block_index].push_front(std::make_pair(ptr->bind_offset, ptr->bind_capacity));
        }
        else
        {
            image_memory_budgets[block_index].push_back(std::make_pair(ptr->bind_offset, ptr->bind_capacity));
        }
    }

    if (!ptr->command_refcount)
    {
        vkDestroyImageView(vkdev->vkdevice(), ptr->imageview, 0);
        vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);

        delete ptr;
    }
}

VkWeightAllocator::VkWeightAllocator(const GPUDevice* _vkdev)
    : VkAllocator(_vkdev)
{
    buffer_offset_alignment = vkdev->info.buffer_offset_alignment;
    bind_memory_offset_alignment = vkdev->info.buffer_image_granularity;

    if (vkdev->info.type == 1)
    {
        // on integrated gpu, there may be device local only memory too, eg. AMD APU
        // assuming larger alignment always keeps us safe :)

        // least common multiple for memory_map_alignment and buffer_offset_alignment and non_coherent_atom_size
        buffer_offset_alignment = least_common_multiple(buffer_offset_alignment, vkdev->info.memory_map_alignment);
        buffer_offset_alignment = least_common_multiple(buffer_offset_alignment, vkdev->info.non_coherent_atom_size);
    }

    block_size = alignSize(8 * 1024 * 1024, buffer_offset_alignment); // 8M
}

VkWeightAllocator::~VkWeightAllocator()
{
    //clear();
    printf("run VkWeightAllocator descontruction function\n");
}

void VkWeightAllocator::clear()
{
    printf("run VkWeightAllocator clear function\n");
}

VkBufferMemory* VkWeightAllocator::fastMalloc(size_t size)
{
    // printf("VkWeightAllocator fastMalloc %lu\n", size);

    size_t aligned_size = alignSize(size, buffer_offset_alignment);

    const int buffer_block_count = buffer_blocks.size();

    // find first spare space in buffer_blocks
    for (int i = 0; i < buffer_block_count; i++)
    {
        size_t free_size = buffer_block_free_spaces[i];
        if (free_size >= aligned_size)
        {
            size_t block_offset = block_size - free_size;
            // return sub buffer
            VkBufferMemory* ptr = new VkBufferMemory;

            ptr->buffer = buffer_blocks[i]->buffer;
            ptr->offset = block_offset;
            ptr->memory = buffer_blocks[i]->memory;
            ptr->capacity = aligned_size;
            ptr->mapped_ptr = buffer_blocks[i]->mapped_ptr;
            ptr->access_flags = 0;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

            buffer_block_free_spaces[i] -= aligned_size;

            return ptr;
        }
    }
    size_t new_block_size = std::max(block_size, aligned_size);

    // create new block
    VkBufferMemory* block = new VkBufferMemory;

    block->buffer = create_buffer(new_block_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    block->offset = 0;

    if (vkdev->info.support_VK_KHR_get_memory_requirements2 && vkdev->info.support_VK_KHR_dedicated_allocation)
    {
        VkBufferMemoryRequirementsInfo2KHR bufferMemoryRequirementsInfo2;
        bufferMemoryRequirementsInfo2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR;
        bufferMemoryRequirementsInfo2.pNext = 0;
        bufferMemoryRequirementsInfo2.buffer = block->buffer;

        VkMemoryRequirements2KHR memoryRequirements2;
        memoryRequirements2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR;
        memoryRequirements2.pNext = 0;

        VkMemoryDedicatedRequirementsKHR memoryDedicatedRequirements;
        memoryDedicatedRequirements.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR;
        memoryDedicatedRequirements.pNext = 0;
        memoryRequirements2.pNext = &memoryDedicatedRequirements;

        vkdev->vkGetBufferMemoryRequirements2KHR(vkdev->vkdevice(), &bufferMemoryRequirementsInfo2, &memoryRequirements2);

        bool dedicatedAllocation = memoryDedicatedRequirements.requiresDedicatedAllocation || memoryDedicatedRequirements.prefersDedicatedAllocation;

        if (dedicatedAllocation)
        {
            // setup memory type and alignment
            if (buffer_memory_type_index == (uint32_t)-1)
            {
                if (vkdev->info.type == 1)
                {
                    // integrated gpu, prefer unified memory
                    buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
                }
                else
                {
                    // discrete gpu, device local
                    buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                }

                mappable = vkdev->is_mappable(buffer_memory_type_index);
                coherent = vkdev->is_coherent(buffer_memory_type_index);
            }

            block->memory = allocate_dedicated_memory(memoryRequirements2.memoryRequirements.size, buffer_memory_type_index, 0, block->buffer);
            // ignore memoryRequirements2.memoryRequirements.alignment as we always bind at zero offset
            vkBindBufferMemory(vkdev->vkdevice(), block->buffer, block->memory, 0);

            block->mapped_ptr = 0;
            if (mappable)
            {
                vkMapMemory(vkdev->vkdevice(), block->memory, 0, new_block_size, 0, &block->mapped_ptr);
            }

            dedicated_buffer_blocks.push_back(block);

            // return sub buffer
            VkBufferMemory* ptr = new VkBufferMemory;

            ptr->buffer = block->buffer;
            ptr->offset = 0;
            ptr->memory = block->memory;
            ptr->capacity = new_block_size;
            ptr->mapped_ptr = block->mapped_ptr;
            ptr->access_flags = 0;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

            return ptr;
        }
    }

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), block->buffer, &memoryRequirements);

    // setup memory type and alignment
    if (buffer_memory_type_index == (uint32_t)-1)
    {
        if (vkdev->info.type == 1)
        {
            // integrated gpu, prefer unified memory
            buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
        }
        else
        {
            // discrete gpu, device local
            buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        }

        mappable = vkdev->is_mappable(buffer_memory_type_index);
        coherent = vkdev->is_coherent(buffer_memory_type_index);
    }

    block->memory = allocate_memory(memoryRequirements.size, buffer_memory_type_index);

    // ignore memoryRequirements.alignment as we always bind at zero offset
    vkBindBufferMemory(vkdev->vkdevice(), block->buffer, block->memory, 0);

    //     printf("VkWeightAllocator M %p", block->buffer);
    block->mapped_ptr = 0;
    if (mappable)
    {
        vkMapMemory(vkdev->vkdevice(), block->memory, 0, new_block_size, 0, &block->mapped_ptr);
    }

    buffer_blocks.push_back(block);

    buffer_block_free_spaces.push_back(new_block_size - aligned_size);

    // return sub buffer
    VkBufferMemory* ptr = new VkBufferMemory;

    ptr->buffer = block->buffer;
    ptr->offset = 0;
    ptr->memory = block->memory;
    ptr->capacity = aligned_size;
    ptr->mapped_ptr = block->mapped_ptr;
    ptr->access_flags = 0;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    return ptr;
}

VkImageMemory* VkWeightAllocator::fastMalloc(int dims, int w, int h, int c, size_t elemsize, int elempack)
{
    if (elempack != 1 && elempack != 4 && elempack != 8 && elempack != 16 && elempack != 32 && elempack != 64)
    {
        printf("elempack must be 1 4 8 16 32 64\n");
        return 0;
    }

    // resolve format
    VkFormat format = VK_FORMAT_UNDEFINED;

    if (elemsize / elempack == 4)
    {
        // fp32
        if (elempack == 1) format = VK_FORMAT_R32_SFLOAT;
        if (elempack == 4) format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (elempack == 8) format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (elempack == 16) format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (elempack == 32) format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (elempack == 64) format = VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    if (elemsize / elempack == 2)
    {
        // fp16
        if (elempack == 1) format = VK_FORMAT_R16_SFLOAT;
        if (elempack == 4) format = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (elempack == 8) format = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (elempack == 16) format = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (elempack == 32) format = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (elempack == 64) format = VK_FORMAT_R16G16B16A16_SFLOAT;
    }

    // resolve image width height depth
    int width = w;
    int height = h;
    int depth = c;

    // large elempack spills on image w
    if (elempack == 8) width *= 2;
    if (elempack == 16) width *= 4;
    if (elempack == 32) width *= 8;
    if (elempack == 64) width *= 16;

    VkImageType image_type;
    VkImageViewType imageview_type;
    if (dims == 1)
    {
        image_type = VK_IMAGE_TYPE_1D;
        imageview_type = VK_IMAGE_VIEW_TYPE_1D;

        if (width > (int)vkdev->info.max_image_dimension_1d)
        {
            printf("image dimension too large %d > %d\n", width, (int)vkdev->info.max_image_dimension_1d);
            return 0;
        }
    }
    else if (dims == 2)
    {
        image_type = VK_IMAGE_TYPE_2D;
        imageview_type = VK_IMAGE_VIEW_TYPE_2D;

        if (width > (int)vkdev->info.max_image_dimension_2d || height > (int)vkdev->info.max_image_dimension_2d)
        {
            printf("image dimension too large %d %d > %d \n", width, height, (int)vkdev->info.max_image_dimension_2d);
            return 0;
        }
    }
    else // if (dims == 3)
    {
        image_type = VK_IMAGE_TYPE_3D;
        imageview_type = VK_IMAGE_VIEW_TYPE_3D;

        if (width > (int)vkdev->info.max_image_dimension_3d || height > (int)vkdev->info.max_image_dimension_3d || depth > (int)vkdev->info.max_image_dimension_3d)
        {
            printf("image dimension too large %d %d %d > %d \n", width, height, depth, (int)vkdev->info.max_image_dimension_3d);
            return 0;
        }
    }

    VkImageMemory* ptr = new VkImageMemory;

    ptr->image = create_image(image_type, width, height, depth, format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    ptr->image_type = image_type;
    ptr->imageview_type = imageview_type;
    ptr->width = width;
    ptr->height = height;
    ptr->depth = depth;
    ptr->format = format;

    if (vkdev->info.support_VK_KHR_get_memory_requirements2 && vkdev->info.support_VK_KHR_dedicated_allocation)
    {
        VkImageMemoryRequirementsInfo2KHR imageMemoryRequirementsInfo2;
        imageMemoryRequirementsInfo2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR;
        imageMemoryRequirementsInfo2.pNext = 0;
        imageMemoryRequirementsInfo2.image = ptr->image;

        VkMemoryRequirements2KHR memoryRequirements2;
        memoryRequirements2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR;
        memoryRequirements2.pNext = 0;

        VkMemoryDedicatedRequirementsKHR memoryDedicatedRequirements;
        memoryDedicatedRequirements.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR;
        memoryDedicatedRequirements.pNext = 0;
        memoryRequirements2.pNext = &memoryDedicatedRequirements;

        vkdev->vkGetImageMemoryRequirements2KHR(vkdev->vkdevice(), &imageMemoryRequirementsInfo2, &memoryRequirements2);

        bool dedicatedAllocation = memoryDedicatedRequirements.requiresDedicatedAllocation || memoryDedicatedRequirements.prefersDedicatedAllocation;

        if (dedicatedAllocation)
        {
            // setup memory type and alignment
            if (image_memory_type_index == (uint32_t)-1)
            {
                if (vkdev->info.type == 1)
                {
                    // integrated gpu, prefer unified memory
                    image_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
                }
                else
                {
                    // discrete gpu, device local
                    image_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                }

                mappable = vkdev->is_mappable(image_memory_type_index);
                coherent = vkdev->is_coherent(image_memory_type_index);
            }

            // bind memory
            ptr->memory = allocate_dedicated_memory(memoryRequirements2.memoryRequirements.size, image_memory_type_index, ptr->image, 0);
            ptr->bind_offset = 0;
            ptr->bind_capacity = memoryRequirements2.memoryRequirements.size;

            // ignore memoryRequirements2.memoryRequirements.alignment as we always bind at zero offset
            vkBindImageMemory(vkdev->vkdevice(), ptr->image, ptr->memory, ptr->bind_offset);

            // do not allow host access to optimal tiling image
            ptr->mapped_ptr = 0;

            ptr->imageview = create_imageview(imageview_type, ptr->image, format);

            ptr->access_flags = 0;
            ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            ptr->command_refcount = 0;

            dedicated_image_memory_blocks.push_back(ptr->memory);

            return ptr;
        }
    }

    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(vkdev->vkdevice(), ptr->image, &memoryRequirements);

    const size_t size = memoryRequirements.size;
    const size_t alignment = std::max((size_t)memoryRequirements.alignment, bind_memory_offset_alignment);

    size_t aligned_size = alignSize(size, alignment);

    const int image_memory_block_count = image_memory_blocks.size();

    // find first spare space in buffer_blocks
    for (int i = 0; i < image_memory_block_count; i++)
    {
        // we cannot use image_memory_block_free_spaces[i] directly for base offset alignment
        size_t bind_base_offset = block_size - image_memory_block_free_spaces[i];
        size_t bind_offset = alignSize(bind_base_offset, alignment);
        if (image_memory_block_free_spaces[i] >= aligned_size + (bind_offset - bind_base_offset))
        {
            // bind at memory offset
            ptr->memory = image_memory_blocks[i];
            ptr->bind_offset = bind_offset;
            ptr->bind_capacity = aligned_size;

            vkBindImageMemory(vkdev->vkdevice(), ptr->image, ptr->memory, ptr->bind_offset);

            // do not allow host access to optimal tiling image
            ptr->mapped_ptr = 0;

            ptr->imageview = create_imageview(imageview_type, ptr->image, format);

            ptr->access_flags = 0;
            ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            ptr->command_refcount = 0;

            if (bind_base_offset != bind_offset)
            {
                // NOTE there is small offset inside bind_base_offset and bind_offset
                // adjust ptr->bind_offset and ptr->bind_capacity after vkBindImageMemory
                // so that memory management could be easier
                aligned_size += (bind_offset - bind_base_offset);

                ptr->bind_offset = bind_base_offset;
                ptr->bind_capacity = aligned_size;
            }

            image_memory_block_free_spaces[i] -= aligned_size;

            return ptr;
        }
    }

    // setup memory type and alignment
    if (image_memory_type_index == (uint32_t)-1)
    {
        if (vkdev->info.type == 1)
        {
            // integrated gpu, prefer unified memory
            image_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
        }
        else
        {
            // discrete gpu, device local
            image_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        }

        mappable = vkdev->is_mappable(image_memory_type_index);
        coherent = vkdev->is_coherent(image_memory_type_index);
    }

    // create new block
    size_t new_block_size = std::max(block_size, aligned_size);

    // bind at memory offset
    ptr->memory = allocate_memory(new_block_size, image_memory_type_index);
    ptr->bind_offset = 0;
    ptr->bind_capacity = aligned_size;

    // ignore memoryRequirements2.memoryRequirements.alignment as we always bind at zero offset
    vkBindImageMemory(vkdev->vkdevice(), ptr->image, ptr->memory, ptr->bind_offset);

    // do not allow host access to optimal tiling image
    ptr->mapped_ptr = 0;

    ptr->imageview = create_imageview(imageview_type, ptr->image, format);

    ptr->access_flags = 0;
    ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    ptr->command_refcount = 0;

    image_memory_blocks.push_back(ptr->memory);
    image_memory_block_free_spaces.push_back(new_block_size - aligned_size);

    return ptr;
}

void VkWeightAllocator::fastFree(VkBufferMemory* ptr)
{
    //     TLOG_INFO("VkWeightAllocator F %p", ptr->buffer);

    delete ptr;
}

void VkWeightAllocator::fastFree(VkImageMemory* ptr)
{
    //     TLOG_INFO("VkWeightAllocator F %p", ptr->memory);

    if (!ptr->command_refcount)
    {
        vkDestroyImageView(vkdev->vkdevice(), ptr->imageview, 0);
        vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);

        delete ptr;
    }
}

VkStagingAllocator::VkStagingAllocator(const GPUDevice* _vkdev)
    : VkAllocator(_vkdev)
{
    mappable = true;
    coherent = true;

    size_compare_ratio = 192; // 0.75f * 256
}

VkStagingAllocator::~VkStagingAllocator()
{
    clear();
}

void VkStagingAllocator::clear()
{
    //     TLOG_INFO("VkStagingAllocator %lu", buffer_budgets.size());

    for (std::list<VkBufferMemory*>::iterator it = buffer_budgets.begin(); it != buffer_budgets.end(); it++)
    {
        VkBufferMemory* ptr = *it;

        //         TLOG_INFO("VkStagingAllocator F %p", ptr->buffer);

        vkUnmapMemory(vkdev->vkdevice(), ptr->memory);
        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

        delete ptr;
    }
    buffer_budgets.clear();
}

VkBufferMemory* VkStagingAllocator::fastMalloc(size_t size)
{
    // printf("VkStagingAllocator fastMalloc %lu\n", size);
    // find free budget
    std::list<VkBufferMemory*>::iterator it = buffer_budgets.begin();
    for (; it != buffer_budgets.end(); it++)
    {
        VkBufferMemory* ptr = *it;

        size_t capacity = ptr->capacity;

        // size_compare_ratio ~ 100%
        if (capacity >= size && ((capacity * size_compare_ratio) >> 8) <= size)
        {
            buffer_budgets.erase(it);

            //             TLOG_INFO("VkStagingAllocator M %p %lu reused %lu", ptr->buffer, size, capacity);

            return ptr;
        }
    }

    VkBufferMemory* ptr = new VkBufferMemory;

    ptr->buffer = create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    ptr->offset = 0;

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), ptr->buffer, &memoryRequirements);

    // setup memory type
    if (buffer_memory_type_index == (uint32_t)-1)
    {
        buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    ptr->memory = allocate_memory(memoryRequirements.size, buffer_memory_type_index);

    // ignore memoryRequirements.alignment as we always bind at zero offset
    vkBindBufferMemory(vkdev->vkdevice(), ptr->buffer, ptr->memory, 0);

    ptr->capacity = size;

    vkMapMemory(vkdev->vkdevice(), ptr->memory, 0, size, 0, &ptr->mapped_ptr);

    ptr->access_flags = 0;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    //     TLOG_INFO("VkStagingAllocator M %p %lu", ptr->buffer, size);

    return ptr;
}

VkImageMemory* VkStagingAllocator::fastMalloc(int dims, int w, int h, int c, size_t elemsize, int elempack)
{
    // staging image is mainly used for storing small piece of dynamic parameters
    // we allocate host memory as a fake image, it's simple and good

    const size_t size = w * h * c * elemsize;

    VkImageType image_type;
    VkImageViewType imageview_type;
    if (dims == 1)
    {
        image_type = VK_IMAGE_TYPE_1D;
        imageview_type = VK_IMAGE_VIEW_TYPE_1D;
    }
    else if (dims == 2)
    {
        image_type = VK_IMAGE_TYPE_2D;
        imageview_type = VK_IMAGE_VIEW_TYPE_2D;
    }
    else // if (dims == 3)
    {
        image_type = VK_IMAGE_TYPE_3D;
        imageview_type = VK_IMAGE_VIEW_TYPE_3D;
    }

    VkImageMemory* ptr = new VkImageMemory;

    ptr->image = 0;
    ptr->image_type = image_type;
    ptr->imageview_type = imageview_type;
    ptr->width = w;
    ptr->height = h;
    ptr->depth = c;
    ptr->format = VK_FORMAT_UNDEFINED;
    ptr->memory = 0;
    ptr->bind_offset = 0;
    ptr->bind_capacity = size;

    ptr->mapped_ptr = malloc(size);

    ptr->imageview = 0;

    ptr->access_flags = 0;
    ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ptr->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    ptr->command_refcount = 0;

    //     TLOG_INFO("VkStagingAllocator M %p %d %d %d %d %d", ptr->image, dims, width, height, depth, format);

    return ptr;
}

void VkStagingAllocator::fastFree(VkBufferMemory* ptr)
{
    //     TLOG_INFO("VkStagingAllocator F %p", ptr->buffer);

    // return to buffer_budgets
    buffer_budgets.push_back(ptr);
}

void VkStagingAllocator::fastFree(VkImageMemory* ptr)
{
    //     TLOG_INFO("VkStagingAllocator F %p", ptr->image);

    free(ptr->mapped_ptr);

    delete ptr;
}

VkWeightStagingAllocator::VkWeightStagingAllocator(const GPUDevice* _vkdev)
    : VkAllocator(_vkdev)
{
    mappable = true;
    coherent = true;
}

VkWeightStagingAllocator::~VkWeightStagingAllocator()
{
}

VkBufferMemory* VkWeightStagingAllocator::fastMalloc(size_t size)
{
    printf("VkWeightStagingAllocator fastMalloc %lu\n", size);
    VkBufferMemory* ptr = new VkBufferMemory;

    ptr->buffer = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    ptr->offset = 0;

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), ptr->buffer, &memoryRequirements);

    // setup memory type
    if (buffer_memory_type_index == (uint32_t)-1)
    {
        buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    ptr->memory = allocate_memory(memoryRequirements.size, buffer_memory_type_index);

    // ignore memoryRequirements.alignment as we always bind at zero offset
    vkBindBufferMemory(vkdev->vkdevice(), ptr->buffer, ptr->memory, 0);

    ptr->capacity = size;

    vkMapMemory(vkdev->vkdevice(), ptr->memory, 0, size, 0, &ptr->mapped_ptr);

    ptr->access_flags = 0;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    //     printf("VkWeightStagingAllocator M %p %lu", ptr->buffer, size);

    return ptr;
}

void VkWeightStagingAllocator::fastFree(VkBufferMemory* ptr)
{
    //     TLOG_INFO("VkWeightStagingAllocator F %p", ptr->buffer);

    vkUnmapMemory(vkdev->vkdevice(), ptr->memory);
    vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
    vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

    delete ptr;
}

} // namespace TEngine
