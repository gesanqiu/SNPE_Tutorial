/*
 * Copyright (c) 2012-2022
 * All Rights Reserved by Thundercomm Technology Co., Ltd. and its affiliates.
 * You may not use, copy, distribute, modify, transmit in any form this file
 * except in compliance with THUNDERCOMM in writing by applicable law.
 * Author: qianyong
 * Date: 2021/06/25
 */
#ifndef __TS_ALG_INTERFACE_H__
#define __TS_ALG_INTERFACE_H__  
    
//
// headers included
//
#include <Common.h>

//
// functions
//
extern "C" void* algInit  (const std::string&                        );
extern "C" void* algInit2 (void*, const std::string&                 );
extern "C" bool  algStart (void*                                     );
extern "C" std::shared_ptr<TsJsonObject> 
                 algProc  (void*, const std::shared_ptr<TsGstSample>&);
extern "C" std::shared_ptr<std::vector<std::shared_ptr<TsJsonObject>>> 
                 algProc2 (void*, const std::shared_ptr<std::vector<
                           std::shared_ptr<TsGstSample>>>&           );
extern "C" bool  algCtrl  (void*, const std::string&                 );
extern "C" void  algStop  (void*                                     );
extern "C" void  algFina  (void*                                     );
extern "C" bool  algSetCb (void*, TsPutResult,  void*                );
extern "C" bool  algSetCb2(void*, TsPutResults, void*                );

#endif //__TS_ALG_INTERFACE_H__

