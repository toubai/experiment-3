package com.example.service;

import cn.hutool.core.date.DateUnit;
import cn.hutool.core.date.DateUtil;
import cn.hutool.core.util.ObjectUtil;
import com.example.common.enums.RoleEnum;
import com.example.entity.Account;
import com.example.entity.Cart;
import com.example.entity.Goods;
import com.example.entity.Orders;
import com.example.mapper.CartMapper;
import com.example.mapper.GoodsMapper;
import com.example.mapper.OrdersMapper;
import com.example.utils.TokenUtils;
import com.github.pagehelper.PageHelper;
import com.github.pagehelper.PageInfo;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;
import java.util.Date;
import java.util.List;

/**
 * 订单业务处理
 **/
@Service
public class OrdersService {

    @Resource
    private OrdersMapper ordersMapper;
    @Resource
    private CartMapper cartMapper;
    @Autowired
    private GoodsMapper goodsMapper;

    /**
     * 新增
     */
    public void add(Orders orders) {

        orders.setOrderId(DateUtil.format(new Date(),"yyyyMMddHH:mm:ss"));
        for (Cart cart : orders.getCartData()) {
            Orders dbOrders = new Orders();
            BeanUtils.copyProperties(orders, dbOrders);
            dbOrders.setGoodsId(cart.getGoodsId());
            dbOrders.setBusinessId(cart.getBusinessId());
            dbOrders.setNum(cart.getNum());
            dbOrders.setPrice(cart.getNum() * cart.getGoodsPrice());
            ordersMapper.insert(dbOrders);

            //删除购物车内对应订单
            cartMapper.deleteById(cart.getId());

            Goods goods = goodsMapper.selectById(cart.getGoodsId());
            goods.setCount(goods.getCount() + cart.getNum());
            goodsMapper.updateById(goods);
        }

    }


    /**
     * 删除
     */
    public void deleteById(Integer id) {
        ordersMapper.deleteById(id);
    }

    /**
     * 批量删除
     */
    public void deleteBatch(List<Integer> ids) {
        for (Integer id : ids) {
            ordersMapper.deleteById(id);
        }
    }

    /**
     * 修改
     */
    public void updateById(Orders orders) {
        ordersMapper.updateById(orders);
    }

    /**
     * 根据ID查询
     */
    public Orders selectById(Integer id) {
        return ordersMapper.selectById(id);
    }

    /**
     * 查询所有
     */
    public List<Orders> selectAll(Orders orders) {
        return ordersMapper.selectAll(orders);
    }

    /**
     * 分页查询
     */
    public PageInfo<Orders> selectPage(Orders orders, Integer pageNum, Integer pageSize) {
        Account currentUser = TokenUtils.getCurrentUser();
        if (RoleEnum.USER.name().equals(currentUser.getRole())) {
            orders.setUserId(currentUser.getId());
        }
        if (RoleEnum.BUSINESS.name().equals(currentUser.getRole())) {
            orders.setBusinessId(currentUser.getId());
        }
        PageHelper.startPage(pageNum, pageSize);
        List<Orders> list = ordersMapper.selectAll(orders);
        return PageInfo.of(list);
    }

}