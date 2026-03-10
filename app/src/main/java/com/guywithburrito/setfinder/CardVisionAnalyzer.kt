package com.guywithburrito.setfinder

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.guywithburrito.setfinder.cv.OpenCVFrameProcessor
import com.guywithburrito.setfinder.tracking.SettingsManager
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat

/**
 * Bridge between CameraX and the stateful CardDetector vision engine.
 * 
 * Implements [ImageAnalysis.Analyzer] to receive frames from the camera, 
 * but acts as a producer/consumer bridge that handles format conversion, 
 * rotation, and frame-dropping logic to maintain live performance.
 */
class CardVisionAnalyzer(
    private val detector: CardDetector,
    private val settingsManager: SettingsManager,
    private val scope: CoroutineScope,
    private val frameProcessor: OpenCVFrameProcessor = OpenCVFrameProcessor()
) : ImageAnalysis.Analyzer {

    // Frame Buffer & Signal
    private val frameSignal = Channel<Unit>(Channel.CONFLATED)
    private var pendingMat: Mat? = null
    private var rotationDegrees: Int = 0

    init {
        // Consumer Loop: Processes the latest available frame from the buffer.
        scope.launch(Dispatchers.Default) {
            for (signal in frameSignal) {
                consumePendingFrame()
            }
        }
    }

    /**
     * Producer: Entry point for CameraX. Converts [ImageProxy] to [Mat],
     * buffers it, and signals the consumer loop.
     */
    override fun analyze(image: ImageProxy) {
        try {
            val mat = frameProcessor.createMat()
            // Using toBitmap() is an easy way to handle YUV->RGB and padding
            Utils.bitmapToMat(image.toBitmap(), mat)
            
            synchronized(this) {
                pendingMat?.release() // Release the frame we're about to overwrite
                pendingMat = mat
                rotationDegrees = image.imageInfo.rotationDegrees
            }
            frameSignal.trySend(Unit)
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            image.close() // Release the ImageProxy back to the pool immediately
        }
    }

    /**
     * Consumer: Takes ownership of the buffered Mat and passes it to the detector.
     */
    private fun consumePendingFrame() {
        val (mat, rotation) = synchronized(this) {
            val m = pendingMat
            pendingMat = null // Take ownership
            m to rotationDegrees
        }

        if (mat == null) return

        try {
            val processedMat = rotateIfNeeded(mat, rotation)
            detector.processFrame(processedMat, settingsManager.singleCardMode)
            
            if (processedMat !== mat) processedMat.release()
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            mat.release()
        }
    }

    private fun rotateIfNeeded(mat: Mat, degrees: Int): Mat {
        val rotationCode = when (degrees) {
            90 -> Core.ROTATE_90_CLOCKWISE
            180 -> Core.ROTATE_180
            270 -> Core.ROTATE_90_COUNTERCLOCKWISE
            else -> return mat
        }
        
        val rotated = frameProcessor.createMat()
        Core.rotate(mat, rotated, rotationCode)
        return rotated
    }
}
