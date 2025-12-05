@echo off
chcp 65001
cls

rem ===== 调试：把前几个参数写到 debug_args.txt =====
echo param1=%1 > debug_args.txt
echo param2=%2 >> debug_args.txt
echo param3=%3 >> debug_args.txt
echo param4=%4 >> debug_args.txt
echo param5=%5 >> debug_args.txt
echo param6=%6 >> debug_args.txt
echo param7=%7 >> debug_args.txt
echo param8=%8 >> debug_args.txt

rem ===== 下面保持不变 =====
set engine=%~1
set video_url=%~2
set streamServerUrl=%~3
set skipFrame=%4
set pushFrq=%5

start "" /b python yolov8_detect_tensorrt.py ^
  --engine "%engine%" ^
  --video_url "%video_url%" ^
  --stream_server_url "%streamServerUrl%" ^
  --skip_frame %skipFrame% ^
  --push_frq %pushFrq% > run_yolo_log.txt 2>&1
