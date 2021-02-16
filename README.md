## 執行
    python main.py

## 目標函數
<table align="center">
  <td align="center">
     <a href="https://www.codecogs.com/eqnedit.php?latex=f(x)=\left(&space;\frac{x}{2}&space;\right)^2&space;&plus;&space;\frac{y^3}{x^2}\;&space;x,y\in(1,10)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)=\left(&space;\frac{x}{2}&space;\right)^2&space;&plus;&space;\frac{y^3}{x^2}\;&space;x,y\in(1,10)" title="f(x)=\left( \frac{x}{2} \right)^2 + \frac{y^3}{x^2}\; x,y\in(1,10)" /></a>
  </td>
  <td>
    <img src="https://github.com/YamGhost/BPNN/blob/master/fig/target%20func.png" />
  </td>
</table>

## 神經網路
<table align="center">
    <thead>
        <tr>
            <th colspan="2">網路流程</th>
            <th>架構</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="1" width="10%">
              <p align="center">
                <text>訓練</text>           
              </p>
            </td>
            <td rowspan="1">
              <p align="center">
                <img src="https://github.com/YamGhost/BPNN/blob/master/fig/%E8%A8%93%E7%B7%B4%E6%B5%81%E7%A8%8B.png" />
              </p>
            </td>
            <td rowspan="2" width="50%">   
              <p align="center">
                <img src="https://github.com/YamGhost/BPNN/blob/master/fig/bpnn.png"/>
              </p>
            </td>
        </tr>        
        <tr>
            <td rowspan="1">
              <p align="center">
                <text>測試</text>           
              </p>
            </td>
            <td rowspan="1">
              <p align="center">
                <img src="https://github.com/YamGhost/BPNN/blob/master/fig/%E6%B8%AC%E8%A9%A6%E6%B5%81%E7%A8%8B.png" />
              </p>
            </td>
        </tr>        
    </tbody>
</table>

## 資料集
<table align="center">
    <thead>
        <tr>
            <th>訓練(400點)</th>
            <th>測試(300點)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>
              <p align="center">
                <img src="https://github.com/YamGhost/BPNN/blob/master/fig/training_400.png" />
              </p>
            </td>
            <td>   
              <p align="center">
                <img src="https://github.com/YamGhost/BPNN/blob/master/fig/testing_300.png"/>
              </p>
            </td>
        </tr>
    </tbody>
</table>

## 結果
<p>
  使用3層神經網路輸入到輸出分別為2、45、1個神經元，擬合目標函數訓練資料為400點、測試資料為300點，下圖藍色圓圈為原始資料集(target)、紅色叉叉為神經網路輸出(output)，分別將x、y、z=f(x)畫於圖上，得到誤差MSE和學習率曲線。
</p>
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?MSE=\frac{1}{N}\sum_{i}^{N}\left&space;(&space;target_{i}-output_{i}&space;\right&space;)^2" title="MSE=\frac{1}{N}\sum_{i}^{N}\left ( target_{i}-output_{i} \right )^2" />
</p>
<table align="center">
    <thead>
        <tr>
            <th>訓練(400點)</th>
            <th>測試(300點)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>
              <p align="center">
                <img src="https://github.com/YamGhost/BPNN/blob/master/fig/N45_2_label.png" />
              </p>
            </td>
            <td>   
              <p align="center">
                <img src="https://github.com/YamGhost/BPNN/blob/master/fig/N45_3_label.png"/>
              </p>
            </td>
        </tr>
    </tbody>
        <thead>
        <tr>
            <th>誤差MSE</th>
            <th>學習率</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>
              <p align="center">
                <img src="https://github.com/YamGhost/BPNN/blob/master/fig/N45_1.png" />
              </p>
            </td>
            <td>   
              <p align="center">
                <img src="https://github.com/YamGhost/BPNN/blob/master/fig/N45_4.png"/>
              </p>
            </td>
        </tr>
    </tbody>
</table>
