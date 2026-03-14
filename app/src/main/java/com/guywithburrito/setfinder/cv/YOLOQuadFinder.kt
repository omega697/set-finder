package com.guywithburrito.setfinder.cv

import android.content.Context
import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * YOLOv8-Pose based QuadFinder.
 * Uses a neural network to detect cards and their 4 corners.
 * More robust to lighting and background clutter than traditional CV.
 */
internal class YOLOQuadFinder(
    context: Context,
    modelPath: String = "yolov8n-pose.tflite",
    private val confidenceThreshold: Float = 0.5f
) : QuadFinder {

    private val interpreter: Interpreter = Interpreter(FileUtil.loadMappedFile(context, modelPath))
    private val inputTensor = interpreter.getInputTensor(0)
    private val outputTensor = interpreter.getOutputTensor(0)
    private val inputShape = inputTensor.shape() // [1, 3, 640, 640]
    private val outputShape = outputTensor.shape() // [1, 17, 8400]
    private val isInputUint8 = inputTensor.dataType() == DataType.UINT8
    private val isOutputUint8 = outputTensor.dataType() == DataType.UINT8

    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(inputShape[1], inputShape[2], ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(if (isInputUint8) 0f else 0f, if (isInputUint8) 1f else 255f))
        .build()

    override fun findCandidates(mat: Mat): List<MatOfPoint2f> {
        return findCandidatesFull(mat).map { it.quad }
    }

    override fun findCandidatesFull(mat: Mat): List<CandidateQuad> {
        val bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bitmap)

        val tensorImage = TensorImage(inputTensor.dataType())
        tensorImage.load(bitmap)
        val processedImage = imageProcessor.process(tensorImage)

        val numElements = outputShape[1] * outputShape[2]
        val bytesPerElement = if (isOutputUint8) 1 else 4
        val outputBuffer = ByteBuffer.allocateDirect(numElements * bytesPerElement)
        outputBuffer.order(ByteOrder.nativeOrder())
        interpreter.run(processedImage.buffer, outputBuffer)
        outputBuffer.rewind()

        val rawOutput = FloatArray(numElements)
        if (isOutputUint8) {
            val scale = outputTensor.quantizationParams().scale
            val zeroPoint = outputTensor.quantizationParams().zeroPoint
            for (i in 0 until numElements) {
                val uByte = outputBuffer.get().toInt() and 0xFF
                rawOutput[i] = (uByte - zeroPoint) * scale
            }
        } else {
            outputBuffer.asFloatBuffer().get(rawOutput)
        }

        val candidates = processOutput(rawOutput, mat.width(), mat.height())
        
        bitmap.recycle()
        return candidates
    }

    private fun processOutput(rawOutput: FloatArray, imgWidth: Int, imgHeight: Int): List<CandidateQuad> {
        val numAnchors = outputShape[2] // 8400
        val allCandidates = mutableListOf<CandidateQuad>()

        for (i in 0 until numAnchors) {
            val confidence = rawOutput[4 * numAnchors + i]
            if (confidence < confidenceThreshold) continue

            // Keypoints (4 corners)
            val points = mutableListOf<Point>()
            for (k in 0 until 4) {
                val kx = rawOutput[(5 + k * 3) * numAnchors + i]
                val ky = rawOutput[(5 + k * 3 + 1) * numAnchors + i]
                points.add(Point(kx.toDouble() * imgWidth, ky.toDouble() * imgHeight))
            }

            val quad = MatOfPoint2f(*points.toTypedArray())
            allCandidates.add(CandidateQuad(quad, mutableSetOf(YOLOStrategy)))
        }

        return mergeAndDeduplicate(allCandidates)
    }

    private fun mergeAndDeduplicate(candidates: List<CandidateQuad>): List<CandidateQuad> {
        if (candidates.isEmpty()) return emptyList()
        // Sort by area as a proxy for better fit, but IoU is the primary filter
        val sorted = candidates.sortedByDescending { Imgproc.contourArea(it.quad) }
        val unique = mutableListOf<CandidateQuad>()
        for (cand in sorted) {
            var isDuplicate = false
            for (u in unique) {
                // Tightened IoU threshold from 0.5 to 0.35 to more aggressively merge
                if (cand.quad.calculateIoU(u.quad) > 0.35) {
                    isDuplicate = true; break
                }
            }
            if (!isDuplicate) unique.add(cand)
            if (unique.size >= 25) break
        }
        return unique
    }

    object YOLOStrategy : QuadFindingStrategy {
        override fun toString() = "YOLO"
    }

    fun close() {
        interpreter.close()
    }
}
