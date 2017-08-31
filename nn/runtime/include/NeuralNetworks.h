/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_H
#define ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_H

// TODO Before submitting to NDK, fix all the TODOs in here.

#if __ANDROID_API__ >= __ANDROID_API_O_MR1__

//TODO These may be useful when we broaden the shared memory support
//     but would be available only for system apps.
//#include <android/hardware_buffer.h>
//#include <hardware/gralloc.h>
//#include <android/hidl/memory/1.0/IMemory.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/**
 * Operand types.
 *
 * [TODO: Make sure these are compatible with TensorFlow Lite.]
 */
enum {
    // The following entries are used to declare scalars.
    ANEURALNETWORKS_FLOAT16 = 0,
    ANEURALNETWORKS_FLOAT32 = 1,
    ANEURALNETWORKS_INT8 = 2,
    ANEURALNETWORKS_UINT8 = 3,
    ANEURALNETWORKS_INT16 = 4,
    ANEURALNETWORKS_UINT16 = 5,
    ANEURALNETWORKS_INT32 = 6,
    ANEURALNETWORKS_UINT32 = 7,
    // The following entries are used to declare tensors.
    ANEURALNETWORKS_TENSOR_FLOAT16 = 8,
    ANEURALNETWORKS_TENSOR_FLOAT32 = 9,
    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM = 10,

    ANEURALNETWORKS_NUMBER_DATA_TYPES = 11
};

/**
 * Operation types.
 *
 * [TODO: Make sure these are compatible with TensorFlow Lite.]
 */
enum {
    ANEURALNETWORKS_AVERAGE_POOL = 0,
    ANEURALNETWORKS_CONCATENATION = 1,
    ANEURALNETWORKS_CONV = 2,
    ANEURALNETWORKS_DEPTHWISE_CONV = 3,
    ANEURALNETWORKS_MAX_POOL = 4,
    ANEURALNETWORKS_L2_POOL = 5,
    ANEURALNETWORKS_DEPTH_TO_SPACE = 6,
    ANEURALNETWORKS_SPACE_TO_DEPTH = 7,
    ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION = 8,
    ANEURALNETWORKS_SOFTMAX = 9,
    ANEURALNETWORKS_RESHAPE = 10,
    ANEURALNETWORKS_SPLIT = 11,
    ANEURALNETWORKS_FAKE_QUANT = 12,
    ANEURALNETWORKS_ADD = 13,
    ANEURALNETWORKS_FULLY_CONNECTED = 14,
    ANEURALNETWORKS_CAST = 15,
    ANEURALNETWORKS_MUL = 16,
    ANEURALNETWORKS_L2_NORMALIZATION = 17,
    ANEURALNETWORKS_LOGISTIC = 18,
    ANEURALNETWORKS_RELU = 19,
    ANEURALNETWORKS_RELU6 = 20,
    ANEURALNETWORKS_RELU1 = 21,
    ANEURALNETWORKS_TANH = 22,
    ANEURALNETWORKS_DEQUANTIZE = 23,
    ANEURALNETWORKS_FLOOR = 24,
    ANEURALNETWORKS_GATHER = 25,
    ANEURALNETWORKS_RESIZE_BILINEAR = 26,
    ANEURALNETWORKS_LSH_PROJECTION = 27,
    ANEURALNETWORKS_LSTM = 28,
    ANEURALNETWORKS_SVDF = 29,
    ANEURALNETWORKS_RNN = 30,
    ANEURALNETWORKS_N_GRAM = 31,
    ANEURALNETWORKS_EMBEDDING_LOOKUP = 32,
    ANEURALNETWORKS_HASHTABLE_LOOKUP = 33,

    ANEURALNETWORKS_NUMBER_OPERATION_TYPES = 34
};

/**
 * Request execution preferences.
 */
enum {
    /**
     * Prefer executing the request in a way that minimizes battery drain.
     * This is desirable for requests that will be executed often.
     */
    ANEURALNETWORKS_PREFER_LOW_POWER = 0,
    /**
     * Prefer returning a single answer as fast as possible, even if this causes
     * more power consumption.
     */
    ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER = 1,
    /**
     * Prefer maximizing the throughput of successive frames, for example when
     * processing successive frames coming from the camera.
     */
    ANEURALNETWORKS_PREFER_SUSTAINED_SPEED = 2,

    ANEURALNETWORKS_NUMBER_PREFERENCES = 3
};

/**
 * Result codes.
 */
enum {
    ANEURALNETWORKS_NO_ERROR = 0,
    ANEURALNETWORKS_OUT_OF_MEMORY = 1,
    ANEURALNETWORKS_INCOMPLETE = 2,
    ANEURALNETWORKS_UNEXPECTED_NULL = 3,
    ANEURALNETWORKS_BAD_DATA = 4,
    ANEURALNETWORKS_OP_FAILED = 5,
    ANEURALNETWORKS_NOT_IMPLEMENTED = 6 // TODO remove
};

// The maximum number of operands and operations that a model may have.
const uint32_t MAX_NUMBER_OF_OPERANDS = 0xFFFFFFFE;
const uint32_t MAX_NUMBER_OF_OPERATIONS = 0xFFFFFFFE;

/**
 * ANeuralNetworksMemory is an opaque type that represents shared memory.
 *
 * By using shared memory, a program can efficiently communicate to the
 * runtime and the drivers the weights and other tensors that define a model.
 * See {@Link ANeuralNetworksModel_setOperandValueFromMemory}.
 *
 * An application should typically create one shared memory object that
 * contains every weight and tensor needed to define one or more models.
 *
 * Shared memory can also be used when specifying the input and output
 * arguments of a request.  See calling {@Link ANeuralNetworksRequest_setInputFromMemory}
 * and {@Link ANeuralNetworksRequest_setOutputFromMemory}.
 *
 * Shared memory handles are created by calling {@link ANeuralNetworksMemory_create}
 * and similar functions.
 */
typedef struct ANeuralNetworksMemory ANeuralNetworksMemory;

/**
 * ANeuralNetworksRequest is an opaque type that can be used to apply a machine
 * learning model to a set of inputs.
 *
 * <p>To use:<ul>
 *    <li>Create a new request instance by calling the
 *        {@link ANeuralNetworksRequest_create} function.</li>
 *    <li>Associate data to the model inputs with
 *        {@link ANeuralNetworksRequest_setInput} or
 *        {@Link ANeuralNetworksRequest_setInputFromMemory}.</li>
 *    <li>Associate output buffers to the model outputs with
 *        {@link ANeuralNetworksRequest_setOutput} or
 *        {@Link ANeuralNetworksRequest_setOutputFromMemory}.</li>
 *    <li>Apply the model with {@link ANeuralNetworksRequest_startCompute}.</li>
 *    <li>Wait for the request to complete with {@link
 * ANeuralNetworksRequest_wait}.</li> <li>Repeat the previous steps as often as
 * needed.</li> <li>Destroy the request with {@link
 * ANeuralNetworksRequest_free}.</li></ul></p>
 *
 * <p>A request can be reused by simply modifying the content of the input
 * buffers and restarting the computation. It's also valid to call
 * ANeuralNetworksRequest_setInput or ANeuralNetworksRequest_setOutput before
 * restarting the request, as long as only the address of the buffer
 * changes.</p>
 *
 * <p>The functions that manipulate requests are thread safe.</p>
 * [TODO: We could have it that it's the responsibility of the application to
 * ensure that no two threads manipulate the same request concurrently. Internal
 * structures not specific to a request would always be protected.]
 */
typedef struct ANeuralNetworksRequest ANeuralNetworksRequest;

/**
 * ANeuralNetworksModel is an opaque type that contains a description of the
 * mathematical operations that constitute the model.
 *
 * <p>The model will be built by calling<ul>
 * <li>{@link ANeuralNetworksModel_create},</li>
 * <li>{@link ANeuralNetworksModel_addOperation},</li>
 * <li>{@link ANeuralNetworksModel_addOperand},</li>
 * </ul>
 *
 * A model is destroyed by calling{@link ANeuralNetworksModel_free}.
 */
typedef struct ANeuralNetworksModel ANeuralNetworksModel;

typedef struct ANeuralNetworksIntList {
    uint32_t count;
    const uint32_t* data;
} ANeuralNetworksIntList;

/**
 * ANeuralNetworksOperandType describes the type of an operand.
 * This structure is used to describe both scalars and tensors.
 */
typedef struct ANeuralNetworksOperandType {
    // The data type, e.g ANEURALNETWORKS_INT8.
    uint32_t type;
    // Count and size of each dimension.  The count should be 0 for scalars.
    ANeuralNetworksIntList dimensions;
    /* These two fields are only used for quantized tensors.
     * They should be zero for scalars and non-fixed point tensors.
     * The dequantized value of each entry is (value - offset) * scale.
     * TODO: revisit once we have a final representation for quantization.
     */
    float scale;
    int32_t offset;
} ANeuralNetworksOperandType;

/**
 * ANeuralNetworksEvent is an opaque type that represents an event
 * that will be signaled once a request completes.
 */
typedef struct ANeuralNetworksEvent ANeuralNetworksEvent;

typedef uint32_t ANeuralNetworksOperationType;

/**
 * Initializes the machine learning runtime.
 *
 * This should be called before any other ANeuralNetworks functions.
 * This function may start work threads, may clean up part of the
 * cache, and query the capabilities of the drivers.
 *
 * As the initialization may take some time, you may want to call
 * this function outside of the initialization path of your application,
 * so that your application starts quickly. [TODO verify the startup cost]
 *
 * Your application should call {@link ANeuralNetworksShutdown} to tear
 * down the runtime.
 *
 * It is safe for a process to call this function multiple times.
 * The first call performs the initialization. Successive calls increase
 * an internal reference count. An equivalent number of calls to
 * ANeuralNetworksShutdown must be performed for the runtime to be
 * destroyed. This enables libraries to safely call Initialize and Shutdown.
 *
 * This function is thread safe.
 *
 * @return NO_ERROR if successful, else [?]
 */
int ANeuralNetworksInitialize();

/**
 * Destroys the machine learning runtime.
 *
 * This function frees any resource used by the runtime. It will wait
 * until in flight requests have completed and will prevent new ones
 * from being started with {@link ANeuralNetworksRequest_startCompute}.
 *
 * Threads blocked on {@link ANeuralNetworksRequest_wait} calls will be
 * released before this function terminates.
 *
 * See {@link ANeuralNetworksInitialize} for details on how multiple calls
 * to Initialize and Shutdown work.
 *
 * This function is thread safe.
 *
 * [TODO It's possible that the Initialize and Shutdown calls don't need to
 *  affect the models created by the ANeuralNetworksModel_* APIs.  If so,
 *  we may want to modify the name of this API and specify it here.]
 */
void ANeuralNetworksShutdown();

/**
 * Creates a shared memory object.
 *
 * Creates a shared memory region of the specified size in bytes.
 * See {@link ANeuralNetworksMemory} for a description on how to use
 * this shared memory.
 */
int ANeuralNetworksMemory_create(size_t size, ANeuralNetworksMemory** memory);

/* TODO Should we also have from Surface, IONBuffer, ashmem and:
int ANeuralNetworksMemory_createFromHidlMemory(android::hardware::hidl_memory hidlMemory,
                                               ANeuralNetworksMemory** memory);
int ANeuralNetworksMemory_createFromFd(int fd, ANeuralNetworksMemory** memory);
int ANeuralNetworksMemory_createFromGrallocBuffer(buffer_handle_t buffer,
                                                  ANeuralNetworksMemory** memory);
int ANeuralNetworksMemory_createFromHardwareBuffer(AHardwareBuffer* buffer,
                                                   ANeuralNetworksMemory** memory);
*/

/**
 * Returns a pointer to the content of the shared memory.
 *
 * Returns a pointer to the shared memory created by {@link ANeuralNetworksMemory_create}.
 */
uint8_t* ANeuralNetworksMemory_getPointer(ANeuralNetworksMemory* memory);


/**
 * Delete a shared memory object.
 *
 * Destroys the object used by the run time to keep track of the shared memory.
 * This will free the underlying actual shared memory if no other code has open
 * handles to this memory.  [TODO verify]
 */
void ANeuralNetworksMemory_free(ANeuralNetworksMemory* memory);

/**
 * Create an empty {@link ANeuralNetworksModel}.
 *
 * <p>This only creates the object.  Computation is performed once
 * {@link ANeuralNetworksRequest_startCompute} is invoked.
 *
 * The model should be constructed with calls to
 * {@link ANeuralNetworksModel_addOperation} and
 * {@link ANeuralNetworksModel_addOperand}
 *
 * <p>{@link ANeuralNetworksModel_free} should be called once the model
 * is no longer needed.</p>
 *
 * This function is thread safe.
 *
 * @param model The {@link ANeuralNetworksModel} to be created.
 *              Set to NULL if unsuccessful.
 *
 * @return NO_ERROR if successful, [?] otherwise.
 */
int ANeuralNetworksModel_create(ANeuralNetworksModel** model);

/**
 * Destroy a model.
 *
 * An application is responsible to make sure that no other thread uses
 * the model at the same time.
 *
 * @param model The model to be destroyed. Passing NULL is acceptable and
 *              results in no operation.
 */
void ANeuralNetworksModel_free(ANeuralNetworksModel* model);

/**
 * Add an operand to a model.
 *
 * The order in which the operands are added is important. The first one added
 * to a model will have the index value 0, the second 1, etc.  These indexes are
 * used as operand identifiers in {@link ANeuralNetworksModel_addOperation},
 * {@link ANeuralNetworksRequest_setInput},
 * {@Link ANeuralNetworksRequest_setInputFromMemory},
 * {@link ANeuralNetworksRequest_setOutput},
 * {@Link ANeuralNetworksRequest_setOutputFromMemory} and
 * {@link ANeuralNetworksRequest_setOperandValue}.
 *
 * To build a model that can accomodate inputs of various sizes, as you may want
 * to do for a CNN, set the size of the dimensions that will vary at run time to
 * 0. These dimensions will have to be set when the application calls
 * {@link ANeuralNetworksRequest_setInput}.
 *
 * An application is responsible to make sure that no other thread uses
 * the model at the same time.
 *
 * A model can't be modified once a request has been created for it by
 * {@link ANeuralNetworksRequest_create}.
 *
 * @param model The model to be modified.
 * @param type The {@link ANeuralNetworksOperandType} that describes the shape
 * of the operand.
 *
 * @return NO_ERROR if successful, [?] otherwise.
 */
int ANeuralNetworksModel_addOperand(ANeuralNetworksModel* model,
                                    const ANeuralNetworksOperandType* type);

/**
 * Sets an operand to a constant value.
 *
 * This value can't be changed when a request is executed.
 *
 * A model can't be modified once a request has been created for it by
 * {@link ANeuralNetworksRequest_create}.
 */
int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel* model, int32_t index,
                                         const void* buffer, size_t length);

/**
 * Sets an operand to a value stored in shared memory.
 *
 * This value can't be changed when a request is executed.
 *
 * A model can't be modified once a request has been created for it by
 * {@link ANeuralNetworksRequest_create}.
 */
int ANeuralNetworksModel_setOperandValueFromMemory(ANeuralNetworksModel* model, int32_t index,
                                                   const ANeuralNetworksMemory* buffer,
                                                   uint32_t offset, size_t length);

/**
 * Add an operation to a model.
 *
 * @param model The model to be modified.
 * @param type The type of the operation.
 * @param inputs An array of indexes identifying each an operand.
 * @param outputs An array of indexes identifying each an operand.
 * [TODO: Make sure these are compatible with TensorFlow Lite.]
 *
 * The operands specified by inputs and outputs must have been
 * previously added by calls to {@link ANeuralNetworksModel_addOperand}.
 *
 * An application is responsible to make sure that no other thread uses
 * the model at the same time.
 *
 * A model can't be modified once a request has been created for it by
 * {@link ANeuralNetworksRequest_create}.
 *
 * @return NO_ERROR if successful, [?] otherwise.
 */
int ANeuralNetworksModel_addOperation(ANeuralNetworksModel* model,
                                      ANeuralNetworksOperationType type,
                                      ANeuralNetworksIntList* inputs,
                                      ANeuralNetworksIntList* outputs);

/**
 * Specfifies which operands will be the model's inputs and outputs.
 *
 * TODO: Can an operand be used for both input and output?
 *
 * @param model The model to be modified.
 * @param inputs An array of indexes identifying the input operands.
 * @param outputs An array of indexes identifying the output operands.
 *
 * The operands specified by inputs and outputs must have been
 * previously added by calls to {@link ANeuralNetworksModel_addOperand}.
 *
 * A model can't be modified once a request has been created for it by
 * {@link ANeuralNetworksRequest_create}.
 */
int ANeuralNetworksModel_setInputsAndOutputs(ANeuralNetworksModel* model,
                                             ANeuralNetworksIntList* inputs,
                                             ANeuralNetworksIntList* outputs);

/**
 * Create a {@link ANeuralNetworksRequest} to apply the given model.
 * This only creates the object.  Computation is only performed once
 * {@link ANeuralNetworksRequest_startCompute} is invoked.
 *
 * <p>The provided model must outlive the request.</p>
 *
 * This function is thread safe.
 *
 * @param model The {@link ANeuralNetworksModel} to be evaluated.
 * @param request The newly created object or NULL if unsuccessful.
 *
 * @return NO_ERROR if successful, BAD_DATA if the model is invalid.
 */
int ANeuralNetworksRequest_create(ANeuralNetworksModel* model, ANeuralNetworksRequest** request);

/**
 * Destroy a request.
 *
 * <p>If called on a request for which
 * {@link ANeuralNetworksRequest_startCompute} has been called, the
 * function will return immediately but will mark the request to be deleted
 * once the computation completes. The related {@link ANeuralNetworksEvent}
 * will be signaled but the {link ANeuralNetworksRequest_wait} will return
 * ERROR_DELETED.
 *
 * This function is thread safe.
 *
 * @param request The request to be destroyed. Passing NULL is acceptable and
 *                results in no operation.
 */
void ANeuralNetworksRequest_free(ANeuralNetworksRequest* request);

/**
 * Sets the execution preference.
 *
 * <p>Provides guidance to the runtime when trade-offs are possible.</p>
 *
 * This function is thread safe.
 *
 * @param request The request to be modified.
 * @param preference Either {@link PREFER_LOW_POWER},
 *                  {@link PREFER_SINGLE_FAST_ANSWER}, or
 *                  {@link PREFER_SUSTAINED_SPEED}.
 *
 * @return NO_ERROR if successful.
 */
int ANeuralNetworksRequest_setPreference(ANeuralNetworksRequest* request, uint32_t preference);

/**
 * Associate a user buffer with an input of the model of the
 * {@link ANeuralNetworksRequest}.
 *
 * <p>The provided buffer must outlive the request.</p>
 *
 * This function is thread safe.
 *
 * @param request The request to be modified.
 * @param index The index of the model operand we're associating the input to.
 * @param type The type of the operand. This is useful if the model did not
 * fully specify the operand. If specified in the model, type should be NULL or
 *             have the same value as specified in the model.
 *             [TODO: We know the dimensions may change.  Anything else?  Base
 * type?]
 * @param buffer The buffer containing the data.
 * @param length The length in bytes of the buffer.
 *
 * @return NO_ERROR if successful, BAD_DATA if the name is not recognized
 *         or the buffer is too small for the input.
 */
int ANeuralNetworksRequest_setInput(ANeuralNetworksRequest* request, int32_t index,
                                    const ANeuralNetworksOperandType* type, const void* buffer,
                                    size_t length);

/**
 * Associate part of a shared memory with an input of the model of the
 * {@link ANeuralNetworksRequest}.
 *
 * <p>The provided shared memory must outlive the request.</p>
 *
 * This function is thread safe.
 *
 * @param request The request to be modified.
 * @param index The index of the model operand we're associating the input to.
 * @param type The type of the operand. This is useful if the model did not
 * fully specify the operand. If specified in the model, type should be NULL or
 *             have the same value as specified in the model.
 *             [TODO: We know the dimensions may change.  Anything else?  Base
 * type?]
 * @param memory The shared memory containing the data.
 * @param offset This specifies the location of the data whithin the shared memory.
 *               The offset is in bytes from the start of shared memory.
 *
 * @return NO_ERROR if successful, BAD_DATA if the name is not recognized
 *         or the buffer is too small for the input.
 */
int ANeuralNetworksRequest_setInputFromMemory(ANeuralNetworksRequest* request, int32_t index,
                                              const ANeuralNetworksOperandType* type,
                                              const ANeuralNetworksMemory* memory, uint32_t offset,
                                              uint32_t length);

/**
 * Associate a user buffer with an output of the model of the
 * {@link ANeuralNetworksRequest}.
 *
 * <p>The provided buffer must outlive the request.</p>
 *
 * This function is thread safe.
 *
 * @param request The request to be modified.
 * @param index The index of the model operand we're associating the input to.
 * @param type The type of the operand. This is useful if the model did not
 * fully specify the operand. If specified in the model, type should be NULL or
 *             have the same value as specified in the model.
 *             [TODO: We know the dimensions may change.  Anything else?  Base
 * type?]
 * @param buffer The buffer where the data will be written.
 * @param length The length in bytes of the buffer.
 *
 * @return NO_ERROR if successful, BAD_DATA if the name is not recognized
 *         or the buffer is too small for the output.
 */
int ANeuralNetworksRequest_setOutput(ANeuralNetworksRequest* request, int32_t index,
                                     const ANeuralNetworksOperandType* type, void* buffer,
                                     size_t length);

/**
 * Associate part of a shared memory with an output of the model of the
 * {@link ANeuralNetworksRequest}.
 *
 * <p>The provided shared memory must outlive the request.</p>
 *
 * @param request The request to be modified.
 * @param index The index of the model operand we're associating the input to.
 * @param type The type of the operand. This is useful if the model did not
 * fully specify the operand. If specified in the model, type should be NULL or
 *             have the same value as specified in the model.
 *             [TODO: We know the dimensions may change.  Anything else?  Base
 * type?]
 * @param offset This specifies the location of the data whithin the shared memory.
 *               The offset is in bytes from the start of shared memory.
 * [todo Would it be useful to have a rect param?]
 *
 * @return NO_ERROR if successful, BAD_DATA if the name is not recognized
 *         or the buffer is too small for the output.
 */
int ANeuralNetworksRequest_setOutputFromMemory(ANeuralNetworksRequest* request, int32_t index,
                                               const ANeuralNetworksOperandType* type,
                                               const ANeuralNetworksMemory* memory, uint32_t offset,
                                               uint32_t length);

/**
 * Queue the request for execution.
 *
 * <p>Puts the request in a queue for execution. Once the model has been
 * applied and the outputs are ready to be consumed, the returned event will be
 * signaled. Use {@link ANeuralNetworksRequest_wait} to wait for that event.
 * </p>
 *
 * Multiple requests can be queued and executed concurrently. The runtime makes
 * no guarantee on the ordering of the completion of the requests.  If it's
 * important to the application, the application should enforces the ordering by
 * using the return events.
 *
 * ANeuralNetworksRequest_wait must be called to recuperate the resources used
 * by the event.
 *
 * This function is thread safe.
 *
 * @param request The request to be modified.
 * @param event The event that will be signaled on completion.
 *              [TODO define the functions to create/delete events]
 *
 * @return NO_ERROR if successful, BAD_DATA if callback is NULL.
 */
int ANeuralNetworksRequest_startCompute(ANeuralNetworksRequest* request,
                                        ANeuralNetworksEvent** event);

/**
 * Waits until the request completes.
 *
 * More than one thread can wait on an event.  When the request completes,
 * all threads will be released.
 * [TODO Should we free just one to enable thread pools?]
 *
 * This function is thread safe.
 *
 * @return NO_ERROR if the request completed normally.
 */
int ANeuralNetworksEvent_wait(ANeuralNetworksEvent* event);

/**
 * Destroys the event.
 *
 * TODO: Figure out lifetime management if multiple threads can wait on an
 * event.
 */
void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event);

__END_DECLS

#endif //  __ANDROID_API__ >= 27

#endif // ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_H
