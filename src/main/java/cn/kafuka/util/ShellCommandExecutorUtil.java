package cn.kafuka.util;

import lombok.extern.slf4j.Slf4j;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
public class ShellCommandExecutorUtil {

    /**
     * (1).通过Process调用cmd或者shell命令
     */
    /*public static Map<String,Object> callProcess(String wordDir, List<String> cmdList) {

        Map<String,Object> resultMap = new HashMap<>();


        try {
            //(1).初始化processBuilder
            ProcessBuilder processBuilder = new ProcessBuilder();

            //(2).添加命令的运行目录
            processBuilder.directory(new File(wordDir));

            //(3).添加执行的命令
            processBuilder.command(cmdList);

            //(4).设置cuda TensorRT的环境变量，供Shell脚本中使用
            String env_var = "/usr/local/cuda-11.8/lib64:/gsis_ai/nvidia/TensorRT-8.5.1.7/lib";
            processBuilder.environment().put("LD_LIBRARY_PATH", env_var);
            log.info("step1 ---> env_var: {}, wordDir: {}, shell/cmd: {}",env_var,wordDir,String.join(" ",cmdList));

            //(5).执行命令
            processBuilder.redirectErrorStream(true);
            Process process = processBuilder.start();
            log.info("step1.5 ---> 即将开始读取子进程输出...");

            StringBuilder output = new StringBuilder();
            String os = System.getProperty("os.name").toLowerCase();
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            if (os.contains("win")){
                reader = new BufferedReader(new InputStreamReader(process.getInputStream(), "GBK"));
            }
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                log.info("step2 ---> 调用shell/cmd成功,code:{},msg:{}",exitCode,output);
            } else {
                log.info("step2 ---> 调用shell/cmd失败,code:{},msg:{}",exitCode,output);
            }
            resultMap.put("code",exitCode);
            resultMap.put("msg",output.toString());
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }

        return resultMap;
    }*/


    /**
     * Windows 专用：通过 ProcessBuilder 启动 cmd / bat，并追加 CUDA+TensorRT 的 bin 到 PATH
     */
    public static Map<String, Object> callProcess(String wordDir, List<String> cmdList) {
        Map<String, Object> resultMap = new HashMap<>();
        try {
            ProcessBuilder pb = new ProcessBuilder();
            pb.directory(new File(wordDir));
            pb.command(cmdList);

            /* ==========  仅 Windows：把 dll 目录追加到 PATH  ========== */
            String cudaBin = "D:\\Data_cuda\\Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin";
            String trtBin  = "D:\\tensorRT\\TensorRT-10.0.0.6\\bin";
            String oldPath = pb.environment().get("PATH");

            /**
             * 新增
             */

            // 先取系统 PATH
            String sysPath = System.getenv("PATH");

// 构造新的 PATH：系统 PATH + CUDA + TensorRT
            StringBuilder newPathBuilder = new StringBuilder();
            if (sysPath != null && !sysPath.isEmpty()) {
                newPathBuilder.append(sysPath);
            }
            if (cudaBin != null && !cudaBin.isEmpty()) {
                newPathBuilder.append(";").append(cudaBin);
            }
            if (trtBin != null && !trtBin.isEmpty()) {
                newPathBuilder.append(";").append(trtBin);
            }

            String newPath = newPathBuilder.toString();
            pb.environment().put("PATH", newPath);

            log.info("step1 ---> PATH: {}, wordDir: {}, shell/cmd: {}",
                    newPath, wordDir, String.join(" ", cmdList));

            /**
             * 结束
             */


            /*String newPath = cudaBin + ";" + trtBin + ";" + oldPath;
            pb.environment().put("PATH", newPath);
            */


            /* 日志：一次性打印路径、工作目录、命令 */
            log.info("step1 ---> PATH: {}, wordDir: {}, shell/cmd: {}", newPath, wordDir, String.join(" ", cmdList));

            /* 合并错误流 -> 只读一份输出即可 */
            pb.redirectErrorStream(true);
            Process process = pb.start();

            /* 异步消费缓冲区，防止阻塞 */
            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), "GBK"))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\n");
                }
            }

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                log.info("step2 ---> 调用成功, code:{}, msg:{}", exitCode, output);
            } else {
                log.warn("step2 ---> 调用失败, code:{}, msg:{}", exitCode, output);
            }
            resultMap.put("code", exitCode);
            resultMap.put("msg", output.toString());
        } catch (Exception e) {
            log.error("执行命令异常", e);
            resultMap.put("code", -1);
            resultMap.put("msg", e.getMessage());
        }
        return resultMap;
    }

    /**
     * Windows 专用：无阻塞启动外部进程（用于长期运行的推流/推理脚本）
     * 只负责启动脚本，不等待其退出，适合异步任务
     *
     * @param wordDir 工作目录
     * @param cmdList 命令列表
     * @return code=0 表示成功启动，code=-1 表示启动失败
     */
    public static Map<String, Object> callProcessNoBlockWindows(String wordDir, List<String> cmdList) {
        Map<String, Object> resultMap = new HashMap<>();
        try {
            ProcessBuilder pb = new ProcessBuilder();
            pb.directory(new File(wordDir));
            pb.command(cmdList);

            // 处理 PATH 环境变量
            String cudaBin = "D:\\Data_cuda\\Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin";
            String trtBin = "D:\\tensorRT\\TensorRT-10.0.0.6\\bin";
            String oldPath = pb.environment().get("PATH");

            // 防止 oldPath 为 null 导致拼接出 "null" 字符串
            String newPath;
            if (oldPath != null && !oldPath.isEmpty()) {
                newPath = cudaBin + ";" + trtBin + ";" + oldPath;
            } else {
                newPath = cudaBin + ";" + trtBin;
            }
            pb.environment().put("PATH", newPath);

            // 重定向错误流
            pb.redirectErrorStream(true);

            // 启动进程（不等待）
            Process process = pb.start();

            log.info("step1.5 (NoBlock) --- Process started successfully, wordDir: {}, shell/cmd: {}",
                    wordDir, String.join(" ", cmdList));

            // 立即返回成功，不读输出、不 waitFor
            resultMap.put("code", 0);
            resultMap.put("msg", "Process started successfully in background");

        } catch (IOException e) {
            log.error("step1.5 (NoBlock) --- Failed to start process", e);
            resultMap.put("code", -1);
            resultMap.put("msg", "Failed to start process: " + e.getMessage());
        } catch (Exception e) {
            log.error("step1.5 (NoBlock) --- Unexpected error", e);
            resultMap.put("code", -1);
            resultMap.put("msg", "Unexpected error: " + e.getMessage());
        }

        return resultMap;
    }

    /**
     * (2).通过Runtime调用cmd或者shell命令
     */
    public static void callRuntime(String wordDir,List<String> cmdList) {

        //(1).设置运行目录
        File workDirFile = new File(wordDir);

        //(2).命令列表
        String[] cmdArr = cmdList.toArray(new String[0]);

        try {
            Process process = Runtime.getRuntime().exec(cmdArr,null,workDirFile);
            process.waitFor();
            int exitValue = process.exitValue();
            if (exitValue != 0) {
                BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
                String line;
                while ((line = errorReader.readLine()) != null) {
                    System.out.println(line);
                }
                errorReader.close();
            }else {
                System.out.println("执行返回码:"+exitValue);
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }
}
