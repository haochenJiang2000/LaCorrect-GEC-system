<template>
  <div>
    &nbsp
  </div>
  <el-row class="tac">
    <el-col :span="30">
      <el-menu
        class="el-menu-vertical-demo"
        default-active="2"
        router
        @open="handleOpen"
        @close="handleClose"
      >
        <el-menu-item index="/submit">
          <span>论文提交</span>
        </el-menu-item>
        <div>&nbsp;</div>
        <el-menu-item index="/test">
          <span>纠错测试</span>
        </el-menu-item>
        <div>&nbsp;</div>
        <el-menu-item >
          <span>
            <el-link href="https://docs.qq.com/doc/DVmZEZ0hUUWNDSVhp" :underline="false">用户手册</el-link>
          </span>
        </el-menu-item>
        <div>&nbsp;</div>
        <!-- <el-menu-item index="/privacy">
          <span>隐私声明</span>
        </el-menu-item>
        <div>&nbsp;</div> -->
        <el-menu-item index="/about">
          <span>关于</span>
        </el-menu-item>
        <div>&nbsp;</div>
      </el-menu>
    </el-col>
  </el-row>
</template>

<script lang="ts" setup>
  const handleOpen = (key: string, keyPath: string[]) => {
    console.log(key, keyPath)
  }
  const handleClose = (key: string, keyPath: string[]) => {
    console.log(key, keyPath)
  }
</script>

  